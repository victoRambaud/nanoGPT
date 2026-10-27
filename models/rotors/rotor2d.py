import torch
import torch.nn as nn
import torch_parallel_scan as tps

from models.transformer_utils import TransformerConfig


def se2_motion_regularization(
    motion: torch.Tensor,
    head_angle: torch.Tensor,
) -> torch.Tensor:

    translation = motion.pow(2).sum(dim=-1)
    rotation = 4.0 * (1.0 - torch.cos(head_angle))

    return (translation + rotation)#.sum(dim=-1).mean()


def reference_compose(x: torch.Tensor, y: torch.Tensor):
    """
    Composition of affine reference updates:

        c -> a * c + b

    x, y: [..., 2], containing (a, b).
    """

    a1, b1 = x[..., :1], x[..., 1:]
    a2, b2 = y[..., :1], y[..., 1:]

    return torch.cat([
        a2 * a1,
        a2 * b1 + b2,
    ], dim=-1)


class Rotor2D(nn.Module):
    """
    Input-dependent SO(2) rotations with a learned reference frame.

    Global frame:
        angle_pi[t] = sum_s<=t angular_velocity[s]

    Reference frame:
        reference_angle[t] = last saved global angle

    Working frame:
        working_angle[t] = angle_pi[t] - reference_angle[t]

    The working frame is used for both:
        - Egocentric-to-allocentric motion in N
        - Rotating attention values

    Returns:
        allocentric_velocity: [B, L, H, K]
        value_rotation: (cos, sin), each [B, L, H, K]
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        self.n_heads = config.n_head
        self.n_blocks = config.n_diag_blocks

        self.reset_frames = config.reset_frames

        assert config.dt_rank == 2, "Rotor2D requires dt_rank == 2"

        # Predict angular velocity in so(2)
        self.angle_proj = nn.Linear(
            config.n_embd,
            self.n_heads,
            bias=False,
        )

        # Predict egocentric velocity in R^2
        self.allocentric_velocity = nn.Linear(
            config.n_embd,
            self.n_heads * 2,
            bias=False,
        )

        # Project allocentric motion into N
        self.out_proj = nn.Linear(
            2, self.n_blocks, bias=False
        )

        # Learnable log-frequencies, initialized at log(1) = 0
        if config.rotate_values and config.freq_grad:
            self.log_freq = nn.Parameter(
                torch.zeros(self.n_heads, self.n_blocks)
            )

        if config.gate_algebra:
            self.allo_gate = nn.Sequential(
                nn.Linear(config.n_embd, self.n_heads),
                nn.Sigmoid(),
            )

            self.value_gate = nn.Sequential(
                nn.Linear(config.n_embd, self.n_heads),
                nn.Sigmoid(),
            )

        if self.reset_frames:
            self.origin_gate = nn.Sequential(
                nn.Linear(config.n_embd, self.n_heads),
                nn.Sigmoid(),
            )

            # Initially preserve the original global reference
            # nn.init.zeros_(self.origin_gate[0].weight)
            # nn.init.constant_(self.origin_gate[0].bias, -2.0)

    def forward(self, x: torch.Tensor):
        b, l, _ = x.shape

        movement_magnitude = self.allocentric_velocity(x).view(
            b, l, self.n_heads, 2
        )
        angular_velocity = self.angle_proj(x).view(
            b, l, self.n_heads
        )
        
        angle_pi = angular_velocity.cumsum(dim=1)
        reference_angle = torch.zeros_like(angle_pi)
        origin_gate = None

        if self.reset_frames:
            p_origin = self.origin_gate(x)
            g_origin = (p_origin > 0.5)

            # Token positions: [B, L, H]
            positions = torch.arange(
                l, device=x.device
            ).view(1, l, 1).expand(b, l, self.n_heads)

            # Last position at which the origin was reset
            reset_positions = torch.where(
                g_origin,
                positions,
                -1,
            )

            last_reset = reset_positions.cummax(dim=1).values

            # Prepend initial reference angle = 0
            angles = torch.cat([
                torch.zeros_like(angle_pi[:, :1]),
                angle_pi,
            ], dim=1)

            # Reference angle at the most recent reset
            reference_angle = angles.gather(
                dim=1,
                index=last_reset + 1,
            )

        # Working frame
        working_angle = angle_pi - reference_angle

        if self.config.gate_algebra:
            allo_gate = self.allo_gate(x)
            value_gate = self.value_gate(x)

            angle_pi_allo = working_angle * allo_gate
            angle_pi_value = working_angle * value_gate

        else:
            angle_pi_allo = working_angle
            angle_pi_value = working_angle

        cos_theta = torch.cos(angle_pi_allo)
        sin_theta = torch.sin(angle_pi_allo)

        vx = movement_magnitude[..., 0]
        vy = movement_magnitude[..., 1]

        allo_velocity = torch.stack(
            [
                vx * cos_theta - vy * sin_theta,
                vx * sin_theta + vy * cos_theta,
            ],
            dim=-1,
        )  # [B, L, H, 2]

        allocentric_velocity = self.out_proj(allo_velocity)

        value_rotation = None

        if self.config.rotate_values:

            if self.config.freq_grad:
                # [1, 1, H, K]
                omega = self.log_freq.exp()[
                    None, None, :, :
                ]

                # [B, L, H, K]
                value_angle = (
                    angle_pi_value.unsqueeze(-1) * omega
                )

            else:
                # [B, L, H, K]
                value_angle = angle_pi_value.unsqueeze(-1).expand(
                    -1, -1, -1, self.n_blocks
                )

            value_rotation = (
                torch.cos(value_angle),
                torch.sin(value_angle),
            )

        velocities_norms = se2_motion_regularization(
            movement_magnitude,
            angular_velocity,
        )

        return allocentric_velocity, {
            "activity_norm_lie_algebra": velocities_norms,
            # "motion_activity_norm": movement_magnitude,
            # "angular_activity_norm": angular_velocity,
            "value_rotation": value_rotation,
            "allo_velocity": allo_velocity,
            "max_angle": working_angle.abs().max().item(),
            "min_angle": working_angle.abs().min().item(),
            "head_angle": angular_velocity,
            "reference_angle": reference_angle,
            "working_angle": working_angle,
            "origin_gate": origin_gate,
        }
