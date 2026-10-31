import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_parallel_scan as tps

from models.transformer_utils import TransformerConfig


def rotvec_to_quaternion(r: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    r: [..., 3]
    returns unit quaternion [..., 4], scalar-first [w, x, y, z].
    """
    theta = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
    half_theta = 0.5 * theta

    # sin(theta/2) / theta
    # Taylor expansion around 0 for stability
    scale = torch.where(
        theta > eps,
        torch.sin(half_theta) / theta,
        0.5 - theta.square() / 48.0,
    )

    w = torch.cos(half_theta)
    xyz = scale * r

    return torch.cat([w, xyz], dim=-1)


def quaternion_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product.
    q1, q2: [..., 4], scalar first.
    """
    w1 = q1[..., :1]
    v1 = q1[..., 1:]

    w2 = q2[..., :1]
    v2 = q2[..., 1:]

    w = w1 * w2 - (v1 * v2).sum(dim=-1, keepdim=True)

    v = (
        w1 * v2
        + w2 * v1
        + torch.cross(v1, v2, dim=-1)
    )

    return torch.cat([w, v], dim=-1)


def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    """
    For unit quaternions.
    """
    return torch.cat(
        [q[..., :1], -q[..., 1:]],
        dim=-1,
    )


def quaternion_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate 3D vectors without constructing rotation matrices.

    q: [..., 4]
    v: [..., 3]
    """
    w = q[..., :1]
    u = q[..., 1:]

    t = 2.0 * torch.cross(u, v, dim=-1)

    return (
        v
        + w * t
        + torch.cross(u, t, dim=-1)
    )


def quaternion_power(q: torch.Tensor, omega: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Raise unit quaternion q to a real-valued power omega.

    q:
        [..., 4]          [w, x, y, z]

    omega:
        broadcastable to q[..., :1]

    If
        q = [cos(alpha), u sin(alpha)]
    then
        q^omega = [cos(omega * alpha),
                   u sin(omega * alpha)]

    where alpha = theta / 2.
    """

    q = q / torch.linalg.vector_norm(
        q, dim=-1, keepdim=True
    ).clamp_min(eps)

    w = q[..., :1]
    v = q[..., 1:]

    # alpha = theta / 2
    # atan2 is generally nicer than acos near |w| ~= 1
    sin_alpha = torch.linalg.vector_norm(
        v, dim=-1, keepdim=True
    )

    alpha = torch.atan2(sin_alpha, w)

    # rotation axis
    axis = v / sin_alpha.clamp_min(eps)

    new_alpha = omega * alpha

    new_w = torch.cos(new_alpha)
    new_v = axis * torch.sin(new_alpha)

    # Near identity, axis is ill-defined but vector part -> 0 anyway.
    new_v = torch.where(
        sin_alpha > eps,
        new_v,
        omega * v,
    )

    return torch.cat([new_w, new_v], dim=-1)


def inv_quaternion(q: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [q[..., :1], -q[..., 1:]],
        dim=-1,
    )


class QuaternionRotor(nn.Module):
    """
    x -> quaternion rotations.

    x:
        [B, L, d_model]

    q:
        [B, L, H, K, 4]

    Each quaternion can rotate one 3D feature block.
    """

    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__()

        self.n_heads = config.n_head
        self.n_blocks = config.n_diag_blocks
        self.config = config

        # Minimal 3-parameter representation of each quaternion
        self.motion_proj = nn.Linear(
            config.n_embd,
            self.n_heads * 3,
            bias=False,
        )

        self.allocentric_velocity = nn.Linear(
            config.n_embd, config.n_head * (config.dt_rank), bias=False
        )

        # self.out_proj = nn.Linear(config.dt_rank, config.n_diag_blocks, bias=False)

        if config.rotate_values and config.freq_grad:
            self.log_freq = nn.Parameter(
                torch.zeros(self.n_heads, self.n_blocks)
            )

        if config.gate_algebra:
            self.allo_gate = nn.Sequential(
                nn.Linear(
                    config.n_embd, self.n_heads, bias=True
                ),
                nn.Sigmoid()
            )

            self.value_gate = nn.Sequential(
                nn.Linear(
                    config.n_embd, self.n_heads, bias=True
                ),
                nn.Sigmoid()
            )

    def rotate_vector(self, quaternion_tensor: torch.Tensor, v: torch.Tensor, mode: str="quat") -> torch.Tensor:
        b, l, h, k, _ = quaternion_tensor.shape
        return quaternion_rotate(quaternion_tensor, v.view(b, l, h, k, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape

        movement_magnitude = self.allocentric_velocity(x).view(
            b, l, self.config.n_head, -1
        )

        # predict rotation vector in IR^3
        base_r = self.motion_proj(x).view(
            b,
            l,
            self.n_heads,
            3,
        )

        # transform rotation vector into a quaternion [B, L, H, 4]
        delta_q = rotvec_to_quaternion(base_r)
        
        q_pi = tps.prefix_scan(
            delta_q,
            prefix_func=quaternion_mul,
            dim=1,
        )   # [B, L, H, 4]

        # Correct numerical drift after the scan
        q_pi = q_pi / torch.linalg.vector_norm(
            q_pi,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        if self.config.gate_algebra:
            allo_gate = self.allo_gate(x).unsqueeze(-1)
            q_pi_allo = quaternion_power(q_pi, allo_gate)

            value_gate = self.value_gate(x).unsqueeze(-1)
            q_pi_value = quaternion_power(q_pi, value_gate)
        else:
            q_pi_allo = q_pi
            q_pi_value = q_pi


        # rotate motion in N = IR^3 by the path-integrated quaternions/head rotations
        allo_velocity = quaternion_rotate(q_pi_allo, movement_magnitude)
        # broadcast to blocks, for the actual motion in N per frequency
        allocentric_velocity = allo_velocity
        # allocentric_velocity = self.out_proj(allo_velocity)
        
        if self.config.rotate_values and self.config.freq_grad:
            omega = self.log_freq.exp()[None, None, :, :, None]
            q = quaternion_power(
                q_pi_value.unsqueeze(-2),
                omega,
            )
        else:
            # Fixed freq = 1: no quaternion power needed.
            q = q_pi_value.unsqueeze(-2).expand(
                -1, -1, -1, self.n_blocks, -1
            )

        # [B, L, H, K, 4]
        rotation_norm = (
            8.0 * delta_q[..., 1:].square().sum(dim=-1)
        )
        return allocentric_velocity, {
            "activity_norm_lie_algebra": rotation_norm,
            "value_rotation": q,
            "allo_velocity": allo_velocity
        }
