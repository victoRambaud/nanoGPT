import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *
from models.transformer_utils import TransformerConfig, init_rotation_matrix
from models.non_commute_rotation_module import EgoNDimEmbedder
from models.rotors.quaternion_rotor import QuaternionRotor, inv_quaternion
from models.rotors.rotor2d import Rotor2D


def se2_motion_regularization(
    motion: torch.Tensor,
    head_angle: torch.Tensor,
) -> torch.Tensor:
    # motion:     (b, l, h, 2)
    # head_angle: (b, l, h)

    translation = motion.pow(2).sum(dim=-1)  # (b, l, h)

    rotation = 4.0 * (1.0 - torch.cos(head_angle))  # (b, l, h)

    reg = translation + rotation  # (b, l, h)

    return reg.sum(dim=-1).mean()


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap radians to (-pi, pi]. Gradient 1 a.e.; kink only at the antipode."""
    return torch.atan2(torch.sin(a), torch.cos(a))


def _smooth_norm(sq_sum: torch.Tensor, eps: float) -> torch.Tensor:
    """sqrt(||.||^2 + eps^2) - eps: ~||.|| away from 0, finite gradient at 0."""
    return torch.sqrt(sq_sum + eps * eps) - eps


def se2_twist_penalty(
    v: torch.Tensor,                    # (B, L, 2) raw translation increments (body frame)
    w: torch.Tensor,                    # (B, L, 1) raw rotation increments (radians)
    *,
    mode: str = "separable",      # "separable" | "joint" | "sparse_group"
    wrap: bool = False,           # False: penalize kinematic effort |w| (a 2pi turn costs 2pi)
                                  # True : penalize resulting group displacement (2pi is free)
    lever_arm=None,               # r-hat; None -> detached RMS of ||v|| (radius-of-gyration proxy)
    rot_weight: float = 1.0,      # extra multiplier on the rotation block
    mu: float = 0.5,              # sparse_group mixing weight
    eps: float = 1e-4,            # norm smoothing / lever-arm floor
    mask: torch.Tensor | None = None,   # (B, L) bool, valid steps
    reduction: str = "mean",      # "mean" | "sum" | "none" -> (B, L)
) -> torch.Tensor:
    """
    separable   : ||v_t||_2 + r*|w_t|            each block hits 0 independently
    joint       : sqrt(||v_t||^2 + r^2 w_t^2)    whole twist zeros at once
    sparse_group: separable + mu * joint         rotations rarer than translations
    """
    v_sq = (v * v).sum(dim=-1)                                # (B, L)
    th = wrap_angle(w) if wrap else w
    absth = th.abs()#.squeeze(-1)                              # (B, L)
    # inertia metric: rotation priced as r * |dtheta| (displacement at radius r)
    if lever_arm is None:
        if mask is not None:
            m = mask.to(v.dtype)
            r = torch.sqrt((v_sq * m).sum() / m.sum().clamp_min(1.0))
        else:
            r = torch.sqrt(v_sq.mean())
        r = r.detach().clamp_min(eps)                         # detach: not gameable
    else:
        r = torch.as_tensor(lever_arm, dtype=v.dtype, device=v.device)

    rot = rot_weight * r * absth                              # (B, L)

    if mode == "separable":
        pen = _smooth_norm(v_sq, eps) + rot
    elif mode == "joint":
        pen = _smooth_norm(v_sq + rot * rot, eps)
    elif mode == "sparse_group":
        pen = _smooth_norm(v_sq, eps) + rot + mu * _smooth_norm(v_sq + rot * rot, eps)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if mask is not None:
        m = mask.to(pen.dtype)
        pen = pen * m
        if reduction == "mean":
            return pen.sum() / m.sum().clamp_min(1.0)
        if reduction == "sum":
            return pen.sum()
        return pen
    if reduction == "mean":
        return pen.mean()
    if reduction == "sum":
        return pen.sum()
    return pen


class ThetaEmbedder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        self.in_proj = nn.Linear(config.n_embd, config.dt_rank * config.n_head)
        self.out_proj = nn.Linear(config.dt_rank, config.n_diag_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        theta_in = self.in_proj(x).view(b, l, self.config.n_head, -1)
        theta_out = self.out_proj(theta_in)
        return theta_out
    

class ThetaEmbedder2(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        self.in_proj = nn.Linear(config.n_embd, config.dt_rank * config.n_head)
        # Parallel projection for all heads: weight shape (n_head, n_diag_blocks, dt_rank)
        self.out_proj = nn.Parameter(
            torch.empty(config.n_head, config.dt_rank, config.n_diag_blocks)
        )
        nn.init.normal_(self.out_proj, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        # (b, l, n_head, dt_rank)
        theta_in = self.in_proj(x).view(b, l, self.config.n_head, -1)
        # einsum: parallel linear over all heads
        # (b, l, n_head, dt_rank) x (n_head, n_diag_blocks, dt_rank) -> (b, l, n_head, n_diag_blocks)
        theta_out = torch.einsum("blhr,hrd->blhd", theta_in, self.out_proj)
        return theta_out


class RotationModule(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        layer_index: Optional[int] = None,
        hiera: bool = False,
    ):
        super().__init__()

        self.config = config

        if config.ego2allo:
            if self.config.dt_rank == 2:
                self.theta_embedd = Rotor2D(config)
                # self.theta_embedd = Ego2AlloEmbedder(config)
            elif self.config.dt_rank == 3:
                self.theta_embedd = QuaternionRotor(config)
            else:
                self.theta_embedd = EgoNDimEmbedder(config=config)

        elif config.shared_inner_theta:
            self.theta_embedd = nn.Sequential(
                nn.Linear(config.n_embd, config.dt_rank),
                nn.Linear(
                    config.dt_rank, config.n_head * config.n_diag_blocks, bias=False
                ),
            )
        else:
            self.theta_embedd = ThetaEmbedder(config)
            # self.theta_embedd = ThetaEmbedder2(config)

        self.theta_act = nn.Identity()

        if config.init_same_head:
            S, freqs = init_rotation_matrix(
                config, layer_index=layer_index, hiera=hiera
            )

            if config.n_approx_steps >= 0:
                self.S = nn.Parameter(S)
            else:
                # (b, l, nh, nb, 1)
                self.freqs = nn.Parameter(
                    freqs.squeeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, config.n_head, 1)
                )
        else:
            head_freqs = list()
            # base_freqs = [config.base_freq / (2**i) for i in range(config.n_head)]
            base_freqs = torch.logspace(
                torch.log2(torch.tensor(config.min_base_freq)),
                torch.log2(torch.tensor(config.base_freq)),
                steps=config.n_head,
                base=2,
            )
            for h in range(self.config.n_head):
                S, freqs = init_rotation_matrix(
                    config, layer_index=layer_index, base_pow=1, base_freq=base_freqs[h]
                )
                head_freqs.append(freqs)
            freqs = torch.stack(head_freqs, dim=0)
            self.freqs = nn.Parameter(freqs.squeeze(-1).unsqueeze(0).unsqueeze(0).log())

        if self.config.dt_rank == 2:
            self.rotation_mode = "sin"
        elif self.config.dt_rank == 3:
            self.rotation_mode = "quat"
        else:
            self.rotation_mode = "mat"

    def forward_torch_exp(self, theta: torch.Tensor) -> torch.Tensor:

        S = self.S - self.S.transpose(-1, -2)
        thetaS = torch.matrix_exp(
            S.view(1, 1, 1, *S.size()) * theta.view(*theta.size(), 1, 1)
        )
        return thetaS, {}

    def mat_inverse(self, rot_dict: Dict):
        if self.rotation_mode == "sin":
            rot_inv = (rot_dict["value_rotation"][0], -rot_dict["value_rotation"][1])
        elif self.rotation_mode == "quat":
            rot_inv = inv_quaternion(rot_dict["value_rotation"])
        else:
            # inverse of a rotation matrix is its transpose since
            # M = exp(S-S.T) => M^-1 = exp(-S+S.T) = exp((S-S.T).T) = M.T
            rot_inv = rot_dict["value_rotation"].transpose(-2, -1)

        return rot_inv

    def forward_sins(self, theta: torch.Tensor):
        # freqs = torch.sqrt(self.freqs**2)
        freqs = self.freqs.exp()
        theta = theta * freqs
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        M = (cos, sin)
        return M, {}

    def rotate_qk(
        self,
        rot_matrix: torch.Tensor,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        mode: str = "sin"
        # self, theta: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate queries and keys by an angle of theta, using exponential of matrix S

        Args:
            rot_matrix (torch.Tensor): Either a true rotation matrix or cos/sin for explicit formulation as in RoPE
            q (torch.Tensor): (b, l, nh, h)
            k (torch.Tensor): (b, l, nh, h)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        b, l, nh, _ = q.shape
        q = q.view(b, l, nh, self.config.n_diag_blocks, self.config.diag_block_size)
        if k is not None:
            k = k.view(b, l, nh, self.config.n_diag_blocks, self.config.diag_block_size)

        if self.rotation_mode == "sin":
            cos = rot_matrix[0].unsqueeze(-1)
            sin = rot_matrix[1].unsqueeze(-1)

            def fast_rotate(x: torch.Tensor) -> torch.Tensor:
                x1 = x[..., 0::2]
                x2 = x[..., 1::2]
                x_rotated_even = x1 * cos - x2 * sin
                x_rotated_odd = x1 * sin + x2 * cos
                x = (
                    torch.stack((x_rotated_even, x_rotated_odd), dim=-1)
                    .flatten(-2)
                    .view(b, l, nh, -1)
                )
                return x

            q = fast_rotate(q)
            if k is not None:
                k = fast_rotate(k)
        elif self.rotation_mode == "quat":
            q = self.theta_embedd.rotate_vector(rot_matrix, q).view(b, l, nh, -1)
            if k is not None:
                k = self.theta_embedd.rotate_vector(rot_matrix, k).view(b, l, nh, -1)
        else:
            q = torch.einsum("blhnij,blhnj->blhni", rot_matrix, q).view(b, l, nh, -1)
            if k is not None:
                k = torch.einsum("blhnij,blhnj->blhni", rot_matrix, k).view(
                    b, l, nh, -1
                )

        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes W_v = mat_exp(sum_h v_h * A_h)
        returns x = W_v*x

        Args:
            x (torch.Tensor): Embedding to transform (b, n, dim_embd) or (b, dim_embd) or (b, ..., n_head, h)
            v (torch.Tensor): Velocity to guide transformation (b, n, dim_v_hidden) or (b, dim_v_hidden)

        Returns:
            torch.Tensor: x multiplied by mat_exp(delta*A) (b, n, dim_embd) or (b, dim_embd)
        """
        t0 = time.time()
        b, l, _ = x.shape
        theta_dict = dict()
        if self.config.shared_inner_theta:
            theta = self.theta_embedd(x).view(
                b, l, self.config.n_head, -1
            )  # b, l, nh, nb
        elif self.config.ego2allo:
            theta, theta_dict = self.theta_embedd(x)
        else:
            theta = self.theta_embedd(x)  # b, l, nh, nb, if theta embedded custom

        theta = self.theta_act(theta)
        thetac = theta.cumsum(dim=1)

        # TODO make sure that method stays stable for large number of steps
        if self.config.n_approx_steps > 0:
            rot_matrix, rot_dict = self.approximate_exp(thetac)
            rot_dict["full_rotcreation"] = time.time() - t0
        if self.config.n_approx_steps == -1:
            rot_matrix, rot_dict = self.forward_sins(thetac)
        else:
            rot_matrix, rot_dict = self.forward_torch_exp(thetac)
        
        rot_dict = {**rot_dict, **theta_dict}
        rot_dict["rot_matrix"] = rot_matrix
        if not self.training:
            rot_dict["theta"] = theta
            rot_dict["thetac"] = thetac
        return rot_matrix, thetac, theta, rot_dict


def decaying_frequencies(max_freq, dim, decay=0.5, device=None):
    """
    Allocate integer frequencies 0..max_freq with exponentially
    decaying multiplicity.

    Args:
        max_freq: maximum integer frequency
        dim: total number of dimensions (must be even for k>0 blocks)
        decay: exponential decay rate

    Returns:
        freqs: tensor of frequencies for each 2D rotation block
        counts: number of blocks per frequency
    """

    device = device or "cpu"

    k = torch.arange(max_freq + 1, device=device)

    # preference weights
    weights = torch.exp(-decay * k)

    # number of 2D blocks available
    n_blocks = dim // 2

    # continuous allocation
    raw = weights / weights.sum() * n_blocks

    # integer allocation
    counts = torch.floor(raw).long()

    # distribute remaining blocks
    remainder = n_blocks - counts.sum()
    if remainder > 0:
        frac = raw - counts.float()
        idx = torch.argsort(frac, descending=True)[:remainder]
        counts[idx] += 1

    # avoid losing frequencies due to rounding
    freqs = torch.repeat_interleave(k, counts)

    return freqs, counts

