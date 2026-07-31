import time
import math

import torch
import torch.nn as nn

from typing import *
from models.transformer_utils import TransformerConfig, init_rotation_matrix
from models.non_commute_rotation_module import EgoNDimEmbedder


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
                self.theta_embedd = Ego2AlloEmbedder(config)
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

    def forward_torch_exp(self, theta: torch.Tensor) -> torch.Tensor:

        S = self.S - self.S.transpose(-1, -2)
        thetaS = torch.matrix_exp(
            S.view(1, 1, 1, *S.size()) * theta.view(*theta.size(), 1, 1)
        )
        return thetaS, {}

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

        if mode == "sin":
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
            mat_exp, rot_dict = self.approximate_exp(thetac)
            rot_dict["full_rotcreation"] = time.time() - t0
        if self.config.n_approx_steps == -1:
            mat_exp, rot_dict = self.forward_sins(thetac)
        else:
            mat_exp, rot_dict = self.forward_torch_exp(thetac)
        
        rot_dict = {**rot_dict, **theta_dict}
        if not self.training and self.config.shared_inner_theta:
            rot_dict["theta_in"] = self.theta_embedd[0](x)
        return mat_exp, thetac, theta, rot_dict


class Ego2AlloEmbedder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.rank = config.dt_rank
        self.n_angles = self.rank * (self.rank - 1) // 2
        self.angular_embed = nn.Linear(
            config.n_embd, config.n_head * (self.n_angles), bias=False
        )
        # self.angle_scale = nn.Parameter(torch.tensor(float(config.angle_scale)), requires_grad=True)
        self.angle_scale = nn.Linear(1, config.n_diag_blocks)
        self.allocentric_velocity = nn.Linear(
            config.n_embd, config.n_head * (self.rank), bias=False
        )
        self.out_proj = nn.Linear(self.rank, config.n_diag_blocks, bias=False)

        if config.rotate_values:
            _, value_freqs = init_rotation_matrix(config, base_freq=config.value_base_freq, follow_rank=False)
            if config.single_freq:
                self.value_freqs = nn.Parameter(
                    torch.ones_like(value_freqs).squeeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, config.n_head, 1),
                    requires_grad=False
                )
            elif config.zero_freqs:
                _, value_freqs = init_rotation_matrix(config, base_freq=config.value_base_freq, follow_rank=False, config_blocks=config.n_diag_blocks//2)
                value_freqs = torch.concat([value_freqs, torch.zeros_like(value_freqs)])
                self.value_freqs = nn.Parameter(
                    value_freqs.squeeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, config.n_head, 1).log(),
                    requires_grad=True
                )
            else:
                self.value_freqs = nn.Parameter(
                    value_freqs.squeeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, config.n_head, 1).log(),
                    requires_grad=True
                )
            # self.value_angle_proj = nn.Linear(1, config.n_diag_blocks)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        b, l, d = x.size()
        raw_angular_velocity = self.angular_embed(x).view(
            b, l, self.config.n_head
        )  # (b, l, n_head * 3)
        # angular_velocity = self.angle_scale * raw_angular_velocity
        angular_velocity = raw_angular_velocity
        movement_magnitude = self.allocentric_velocity(x).view(
            b, l, self.config.n_head, -1
        )  # (b, l, n_head * 3)
        
        # velocities_norms = (4*((angular_velocity/2).sin()**2) + 0.0 * (movement_magnitude.norm(p=2, dim=-1)**2)).sum(dim=-1).mean(dim=1).mean()
        # print(velocities_norms)
        # velocities_norms = (2*(angular_velocity/2).sin().abs()).sum(dim=-1).mean(dim=1).mean()
        velocities_norms = se2_twist_penalty(
            v=movement_magnitude,
            w=angular_velocity,
            reduction=None,
            mode="joint"
        )
        velocities_norms = velocities_norms.sum(dim=-1).mean()
        
        if self.config.full_ego_path:
            head_angle = angular_velocity.cumsum(dim=1)  # (b, l, n_head)
        else:
            # TODO rotation by angular velocity probably needed after retrieval
            head_angle = (
                angular_velocity.cumsum(dim=1) - angular_velocity
            )  # (b, l, n_head)
        head_angle = head_angle % (2 * math.pi)
        cos_theta = torch.cos(head_angle)  # (b, l, n_head)
        sin_theta = torch.sin(head_angle)  # (b, l, n_head)

        vx = movement_magnitude[..., 0]
        vy = movement_magnitude[..., 1]
        
        vx_allo = vx * cos_theta - vy * sin_theta
        vy_allo = vx * sin_theta + vy * cos_theta

        allocentric_velocity = torch.stack(
            [vx_allo, vy_allo], dim=-1
        )  # (b, l, n_head, 2)

        allo_velocity = self.out_proj(allocentric_velocity)

        value_rotation = None
        if self.config.rotate_values:
            if not self.config.full_ego_path:
                head_angle = head_angle + angular_velocity
            head_angle = self.angle_scale(head_angle.unsqueeze(-1))
            cos_head = torch.cos( - self.value_freqs.exp() * head_angle)
            sin_head = torch.sin( - self.value_freqs.exp() * head_angle)
            # cos_head = torch.cos( - self.value_freqs.exp() * head_angle.unsqueeze(-1))
            # sin_head = torch.sin( - self.value_freqs.exp() * head_angle.unsqueeze(-1))
            value_rotation = (cos_head, sin_head)
        return allo_velocity, {
            "activity_norm": velocities_norms,
            "value_rotation": value_rotation,
            "max_angle": head_angle.abs().max().item(),
            "min_angle": head_angle.abs().max().item(),
        }
