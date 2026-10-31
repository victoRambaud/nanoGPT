"""Rank-aware, per-head toroidal positional rotations.

For each attention head h:
    delta_p[t, h] in R^k                         (embedding / ego2allo)
    theta[t, h, m] = <u[h, m], delta_p[t, h]>    (per-head block direction)
    theta_c[t, h, m] = sum_{s<=t} theta[s,h,m]  (path integration)
    phi[t, h, m] = exp(log_freq[h,m]) * theta_c[t,h,m]

Each block m rotates a 2D subspace by phi[t,h,m], yielding a representation
R^k -> SO(2)^M, M = config.n_diag_blocks (not necessarily M == k).

The frame / value rotations produced by the ego2allo modules are distinct
from these positional rotations; they remain in rot_dict['value_rotation'].
"""

import math
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_utils import TransformerConfig, init_rotation_matrix
from models.non_commute_rotation_module import EgoNDimEmbedder
from models.rotors.quaternion_rotor import QuaternionRotor, inv_quaternion
from models.rotors.rotor2d import Rotor2D


def se2_motion_regularization(motion: torch.Tensor, head_angle: torch.Tensor) -> torch.Tensor:
    """Quadratic regularization of 2D translation and angle increments."""
    translation = motion.square().sum(dim=-1)
    rotation = 4.0 * (1.0 - torch.cos(head_angle))
    return (translation + rotation).sum(dim=-1).mean()


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


def _smooth_norm(sq_sum: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(sq_sum + eps * eps) - eps


def se2_twist_penalty(
    v: torch.Tensor,
    w: torch.Tensor,
    *,
    mode: str = "separable",
    wrap: bool = False,
    lever_arm=None,
    rot_weight: float = 1.0,
    mu: float = 0.5,
    eps: float = 1e-4,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Optional translation/rotation regularizer (kept for API compatibility)."""
    v_sq = v.square().sum(dim=-1)
    angle = wrap_angle(w) if wrap else w

    if lever_arm is None:
        if mask is not None:
            m = mask.to(v.dtype)
            r = torch.sqrt((v_sq * m).sum() / m.sum().clamp_min(1.0))
        else:
            r = torch.sqrt(v_sq.mean())
        r = r.detach().clamp_min(eps)
    else:
        r = torch.as_tensor(lever_arm, dtype=v.dtype, device=v.device)

    rot = rot_weight * r * angle.abs()
    if mode == "separable":
        penalty = _smooth_norm(v_sq, eps) + rot
    elif mode == "joint":
        penalty = _smooth_norm(v_sq + rot.square(), eps)
    elif mode == "sparse_group":
        penalty = _smooth_norm(v_sq, eps) + rot + mu * _smooth_norm(v_sq + rot.square(), eps)
    else:
        raise ValueError(f"Unknown twist-penalty mode: {mode!r}")

    if mask is not None:
        m = mask.to(penalty.dtype)
        penalty = penalty * m
        if reduction == "mean":
            return penalty.sum() / m.sum().clamp_min(1.0)
    elif reduction == "mean":
        return penalty.mean()

    if reduction == "sum":
        return penalty.sum()
    if reduction == "none":
        return penalty
    raise ValueError(f"Unknown reduction: {reduction!r}")


def _rank_aware_rope_init(
    n_heads: int,
    n_blocks: int,
    rank: int,
    *,
    base_freq: float,
    init_same_head: bool,
    min_base_freq: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize axis directions and log frequencies for a k-dimensional RoPE.

    Blocks are distributed as evenly as possible across k axes. Along each
    axis, the frequencies follow standard 1D RoPE (base^(-j / n_axis_blocks)).
    Rank 1 therefore gives exactly the standard 1D spectrum. Rank 2 gives
    two independently restarted 1D spectra, etc.

    Returns:
        directions: [H, M, K], initially one-hot on a coordinate axis
        log_freqs: [1, 1, H, M], learnable log-frequency initialization
    """
    if min(n_heads, n_blocks, rank) <= 0:
        raise ValueError("n_heads, n_blocks and rank must all be positive")
    if n_blocks < rank:
        raise ValueError(
            f"Cannot give each of {rank} axes a RoPE block with only "
            f"{n_blocks} blocks. Increase n_diag_blocks or reduce dt_rank."
        )
    if base_freq <= 0:
        raise ValueError("base_freq must be positive")

    if init_same_head:
        head_bases = torch.full((n_heads,), float(base_freq), dtype=torch.float32)
    else:
        if min_base_freq is None or min_base_freq <= 0:
            raise ValueError("min_base_freq must be positive when init_same_head=False")
        head_bases = torch.exp(
            torch.linspace(math.log(float(min_base_freq)), math.log(float(base_freq)), n_heads)
        )

    directions = torch.zeros(n_heads, n_blocks, rank)
    log_freqs = torch.empty(n_heads, n_blocks)
    offset = 0

    for axis in range(rank):
        n_axis_blocks = n_blocks // rank + (axis < n_blocks % rank)
        block_ids = slice(offset, offset + n_axis_blocks)
        directions[:, block_ids, axis] = 1.0

        exponents = torch.arange(n_axis_blocks, dtype=torch.float32) / n_axis_blocks
        log_freqs[:, block_ids] = -head_bases.log()[:, None] * exponents[None, :]
        offset += n_axis_blocks

    return directions, log_freqs[None, None, :, :]


class ThetaEmbedder(nn.Module):
    """Learn head-specific increments in R^k; no block projection here."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_head
        self.rank = config.dt_rank
        self.in_proj = nn.Linear(config.n_embd, self.n_heads * self.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return self.in_proj(x).reshape(b, l, self.n_heads, self.rank)


# Legacy name retained for import compatibility; the two former implementations
# differed only in how they projected R^k to rotation blocks, which is now done
# by PerHeadTorusProjection for every branch.
ThetaEmbedder2 = ThetaEmbedder


class PerHeadTorusProjection(nn.Module):
    """Learn directions from head-specific R^k motion to SO(2)^M blocks.

    Unit-normalizing the directions prevents a pure scale redundancy with
    learned log frequencies. Directions can change freely during training.
    """

    def __init__(self, directions: torch.Tensor):
        super().__init__()
        self.directions = nn.Parameter(directions)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        directions = F.normalize(self.directions, p=2, dim=-1, eps=1e-8)
        return torch.einsum("blhk,hmk->blhm", motion, directions)


class RotationModule(nn.Module):
    """Public API is unchanged: forward returns (rot, thetac, theta, dict)."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_index: Optional[int] = None,
        hiera: bool = False,
    ):
        super().__init__()
        self.config = config

        # Preserve the existing ego2allo selection and its value/frame logic.
        if config.ego2allo:
            if config.dt_rank == 2:
                self.theta_embedd = Rotor2D(config)
            elif config.dt_rank == 3:
                self.theta_embedd = QuaternionRotor(config)
            else:
                self.theta_embedd = EgoNDimEmbedder(config=config)
        else:
            self.theta_embedd = ThetaEmbedder(config)

        self.n_torus_blocks = config.head_dim // 2
        self.torus_block_size = 2
        directions, log_freqs = _rank_aware_rope_init(
            config.n_head,
            self.n_torus_blocks,
            config.dt_rank,
            base_freq=float(config.base_freq),
            init_same_head=config.init_same_head,
            min_base_freq=float(config.min_base_freq) if not config.init_same_head else None,
        )
        self.torus_projection = PerHeadTorusProjection(directions)
        # Kept as self.freqs for compatibility with code inspecting this field.
        # IMPORTANT: self.freqs stores LOG frequencies in *both* init branches.
        self.freqs = nn.Parameter(log_freqs)
        self.theta_act = nn.Identity()

        # Legacy generalized matrix-exponential path retained. The toroidal
        # sin/cos path (n_approx_steps == -1) uses the learned log frequencies.
        if config.n_approx_steps >= 0:
            S, _ = init_rotation_matrix(config, layer_index=layer_index, hiera=hiera)
            self.S = nn.Parameter(S)

        # This identifies the format of ego2allo *value* rotations, rather
        # than assuming the positional RoPE rotations must have this format.
        self.rotation_mode = (
            "sin" if config.dt_rank == 2 else
            "quat" if config.dt_rank == 3 else "mat"
        )

    def _motion_and_aux(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        b, l, _ = x.shape
        if self.config.ego2allo:
            motion, aux = self.theta_embedd(x)
        elif self.config.shared_inner_theta:
            motion = self.theta_embedd(x)[:, :, None, :].expand(
                b, l, self.config.n_head, self.config.dt_rank
            )
            aux = {}
        else:
            motion = self.theta_embedd(x)
            aux = {}

        expected = (b, l, self.config.n_head, self.config.dt_rank)
        if tuple(motion.shape) != expected and self.config.ego2allo:
            # Compatibility with older rotors which return already-projected
            # block angles but expose the underlying R^k motion in the dict.
            underlying_motion = aux.get("allo_velocity")
            if isinstance(underlying_motion, torch.Tensor) and tuple(underlying_motion.shape) == expected:
                motion = underlying_motion

        if tuple(motion.shape) != expected:
            raise ValueError(
                f"The motion embedder must return [B,L,H,K]={expected}; "
                f"got {tuple(motion.shape)}. In an imported ego2allo rotor, "
                "return the R^k allocentric motion BEFORE its old out_proj "
                "to n_diag_blocks, or expose it as aux['allo_velocity']. "
                "RotationModule now does the block projection itself."
            )
        return motion, aux

    def forward_sins(self, theta: torch.Tensor):
        """Convert integrated unscaled block angles to SO(2) rotations."""
        angles = theta * self.freqs.exp()  # [B, L, H, M]
        return (torch.cos(angles), torch.sin(angles)), {}

    def forward_torch_exp(self, theta: torch.Tensor):
        """Legacy generalized matrix-exponential branch."""
        S = self.S - self.S.transpose(-1, -2)
        thetaS = torch.matrix_exp(
            S.view(1, 1, 1, *S.size()) * theta.reshape(*theta.shape, 1, 1)
        )
        return thetaS, {}

    def mat_inverse(self, rot_dict: Dict):
        """Invert value/frame rotation, preserving the prior dict API."""
        value_rotation = rot_dict["value_rotation"]
        if isinstance(value_rotation, (tuple, list)):
            return value_rotation[0], -value_rotation[1]
        if self.rotation_mode == "quat":
            return inv_quaternion(value_rotation)
        return value_rotation.transpose(-2, -1)

    def rotate_qk(
        self,
        rot_matrix,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        mode: str = "sin",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Rotate [B,L,H,head_dim] q (and optional k) by block rotations.

        A (cos,sin) tuple denotes SO(2)^M positional rotations for any rank.
        A quaternion tensor retains the legacy ego2allo 3D value path.
        """
        b, l, nh, _ = q.shape
        q = q.reshape(b, l, nh, self.n_torus_blocks, self.torus_block_size)
        if k is not None:
            k = k.reshape(b, l, nh, self.n_torus_blocks, self.torus_block_size)

        if isinstance(rot_matrix, (tuple, list)):
            cos, sin = (v.unsqueeze(-1) for v in rot_matrix)

            def rotate(x):
                if x.shape[-1] % 2:
                    raise ValueError("SO(2) blocks must have even diag_block_size")
                even, odd = x[..., 0::2], x[..., 1::2]
                return torch.stack(
                    (even * cos - odd * sin, even * sin + odd * cos), dim=-1
                ).flatten(-2).reshape(b, l, nh, -1)

        elif self.rotation_mode == "quat" and rot_matrix.shape[-1] == 4:
            def rotate(x):
                return self.theta_embedd.rotate_vector(rot_matrix, x).reshape(b, l, nh, -1)
        else:
            def rotate(x):
                return torch.einsum("blhnij,blhnj->blhni", rot_matrix, x).reshape(b, l, nh, -1)

        return rotate(q), rotate(k) if k is not None else None

    def forward(self, x: torch.Tensor):
        """Return (rot_matrix, thetac, theta, rot_dict) as before.

        theta: [B,L,H,M] per-token *unscaled* block increments.
        thetac: [B,L,H,M] integrated increments (unscaled).
        rot_matrix: (cos,sin) when n_approx_steps == -1.

        For ego2allo, auxiliary value rotations and other diagnostics are
        passed through unchanged from the imported frame module.
        """
        t0 = time.time()
        motion, theta_dict = self._motion_and_aux(x)
        theta = self.theta_act(self.torus_projection(motion))
        thetac = theta.cumsum(dim=1)

        if self.config.n_approx_steps > 0:
            # Kept for compatibility with projects providing approximate_exp.
            rot_matrix, rot_dict = self.approximate_exp(thetac)
            rot_dict["full_rotcreation"] = time.time() - t0
        elif self.config.n_approx_steps == -1:
            rot_matrix, rot_dict = self.forward_sins(thetac)
        else:
            rot_matrix, rot_dict = self.forward_torch_exp(thetac)

        rot_dict = {**rot_dict, **theta_dict, "rot_matrix": rot_matrix}
        if not self.training:
            rot_dict["theta"] = theta
            rot_dict["thetac"] = thetac
        return rot_matrix, thetac, theta, rot_dict


def decaying_frequencies(max_freq, dim, decay=0.5, device=None):
    """Allocate integer harmonic indices 0..max_freq with decaying counts."""
    k = torch.arange(max_freq + 1, device=device or "cpu")
    weights = torch.exp(-decay * k)
    n_blocks = dim // 2
    raw = weights / weights.sum() * n_blocks
    counts = torch.floor(raw).long()
    remainder = int(n_blocks - counts.sum())
    if remainder:
        idx = torch.argsort(raw - counts.float(), descending=True)[:remainder]
        counts[idx] += 1
    return torch.repeat_interleave(k, counts), counts


class Ego2AlloEmbedder(nn.Module):
    """Legacy standalone SE(2) embedder; now returns R^2 motion, not block angles.

    RotationModule defaults to the imported Rotor2D when dt_rank=2. This
    class is retained for experiments that explicitly select it instead.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        if config.dt_rank != 2:
            raise ValueError("Ego2AlloEmbedder is SE(2)-specific; use QuaternionRotor/EgoNDimEmbedder for k>2")
        self.config = config

        def velocity_linear(out_features):
            proj = nn.Linear(config.n_embd, out_features, bias=False)
            return nn.Sequential(proj, nn.RMSNorm(out_features)) if config.norm_velocities else proj

        self.angular_embed = velocity_linear(config.n_head)
        self.allocentric_velocity = velocity_linear(config.n_head * 2)

        if config.gate_algebra:
            self.head_algebra_gate = nn.Sequential(nn.Linear(config.n_embd, config.n_head, bias=False), nn.Sigmoid())
            self.value_gate = nn.Sequential(nn.Linear(config.n_embd, config.n_head, bias=False), nn.Sigmoid())

        if config.rotate_values:
            M = config.n_diag_blocks
            if config.integer_freqs:
                integer, _ = decaying_frequencies(config.block_max_init, dim=2 * M)
                self.register_buffer("fixed_value_freqs", integer.float().view(1, 1, 1, M))
                self.value_log_freqs = None
            else:
                _, log_freqs = _rank_aware_rope_init(
                    1, M, 1,
                    base_freq=float(config.value_base_freq),
                    init_same_head=True,
                )
                if config.single_freq:
                    log_freqs.zero_()
                self.value_log_freqs = nn.Parameter(
                    log_freqs.expand(1, 1, config.n_head, M).clone(),
                    requires_grad=bool(getattr(config, "freq_grad", True)),
                )
                self.register_buffer(
                    "value_zero_mask",
                    (torch.arange(M) >= M // 2).view(1, 1, 1, M) if config.zero_freqs
                    else torch.zeros(1, 1, 1, M, dtype=torch.bool),
                )

    def _value_frequencies(self):
        if self.config.integer_freqs:
            return self.fixed_value_freqs
        return self.value_log_freqs.exp().masked_fill(self.value_zero_mask, 0.0)

    def forward(self, x: torch.Tensor):
        b, l, _ = x.shape
        angular_velocity = self.angular_embed(x).reshape(b, l, self.config.n_head)
        movement = self.allocentric_velocity(x).reshape(b, l, self.config.n_head, 2)

        if self.config.gate_algebra:
            head_delta = angular_velocity * self.head_algebra_gate(x)
            value_delta = angular_velocity * self.value_gate(x)
        else:
            head_delta = angular_velocity
            value_delta = angular_velocity

        # Orientation used to rotate this step's egocentric translation.
        head_angle = head_delta.cumsum(dim=1)
        if not self.config.full_ego_path:
            head_angle = head_angle - head_delta
        head_angle = torch.remainder(head_angle, 2 * math.pi)

        cos, sin = head_angle.cos(), head_angle.sin()
        vx, vy = movement.unbind(-1)
        allo_motion = torch.stack((vx * cos - vy * sin, vx * sin + vy * cos), dim=-1)

        value_rotation = None
        if self.config.rotate_values:
            value_angle = value_delta.cumsum(dim=1)[..., None]
            value_phase = value_angle * self._value_frequencies()
            value_rotation = (value_phase.cos(), value_phase.sin())

        return allo_motion, {
            "activity_norm": se2_motion_regularization(movement, angular_velocity),
            "value_rotation": value_rotation,
            "allo_velocity": allo_motion,
            "max_angle": head_angle.detach().abs().max().item(),
            "min_angle": head_angle.detach().abs().min().item(),
            "head_angle": angular_velocity,
        }
