
import torch
import torch.nn as nn
import torch_parallel_scan as tps

from models.transformer_utils import TransformerConfig


def se2_motion_regularization(
    motion: torch.Tensor,
    head_angle: torch.Tensor,
    reg_mode: str="quad"
) -> torch.Tensor:
    translation = 0.
    if reg_mode == "sparse":
        rotation = 2.0 * torch.sin(head_angle / 2.0).abs()
    elif reg_mode == "quad":
        rotation = 4.0 * (1.0 - torch.cos(head_angle))
    elif reg_mode == "both":
        rotation = 4.0 * (1.0 - torch.cos(head_angle))
        translation = motion.pow(2).sum(dim=-1)
    return translation + rotation


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
    H1 acts on N and values.

    Optionally, H2 acts on the second half of value blocks,
    while H1 acts on the first half.

    With h2=False, H1 acts on all value blocks.

    Angular increments are gated before path integration.
    All value rotations use fixed unit frequency.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        self.n_heads = config.n_head
        self.n_blocks = config.n_diag_blocks
        self.reset_frames = config.reset_frames

        # H2 only exists when value rotation is enabled.
        self.use_h2 = config.h2 and config.rotate_values

        assert config.dt_rank == 2, "Rotor2D requires dt_rank == 2"

        if self.use_h2:
            assert self.n_blocks % 2 == 0, (
                "Value blocks must split evenly between H1 and H2"
            )

        # H1: angular velocity
        self.angle_proj = nn.Linear(
            config.n_embd,
            self.n_heads,
            bias=False,
        )

        # H2: optional independent angular velocity
        if self.use_h2:
            self.angle_proj_h2 = nn.Linear(
                config.n_embd,
                self.n_heads,
                bias=False,
            )

        # Egocentric velocity in R2
        self.allocentric_velocity = nn.Linear(
            config.n_embd,
            self.n_heads * 2,
            bias=False,
        )

        # Projection into N
        self.out_proj = nn.Linear(
            2,
            self.n_blocks,
            bias=False,
        )

        if config.gate_algebra:
            # H1 gate, acting on both N and V1 (or all V)
            self.allo_gate = nn.Sequential(
                nn.Linear(config.n_embd, self.n_heads),
                nn.Sigmoid(),
            )

            if self.use_h2:
                self.value_gate_h2 = nn.Sequential(
                    nn.Linear(config.n_embd, self.n_heads),
                    nn.Sigmoid(),
                )

        if self.reset_frames:
            self.origin_gate = nn.Sequential(
                nn.Linear(config.n_embd, self.n_heads),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor):
        b, l, _ = x.shape

        movement_magnitude = self.allocentric_velocity(x).view(
            b, l, self.n_heads, 2
        )

        angular_velocity = self.angle_proj(x).view(
            b, l, self.n_heads
        )

        if self.use_h2:
            angular_velocity_h2 = self.angle_proj_h2(x).view(
                b, l, self.n_heads
            )

        # Gate angular increments before integration.
        if self.config.gate_algebra:
            angular_velocity = angular_velocity * self.allo_gate(x)

            if self.use_h2:
                angular_velocity_h2 = (
                    angular_velocity_h2 * self.value_gate_h2(x)
                )

        angle_pi = angular_velocity.float().cumsum(dim=1)
        reference_angle = torch.zeros_like(angle_pi)

        if self.use_h2:
            angle_pi_h2 = angular_velocity_h2.float().cumsum(dim=1)
            reference_angle_h2 = torch.zeros_like(angle_pi_h2)

        origin_gate = None

        if self.reset_frames:
            origin_gate = self.origin_gate(x)
            g_origin = origin_gate > 0.5

            positions = torch.arange(
                l, device=x.device
            ).view(1, l, 1).expand(b, l, self.n_heads)

            reset_positions = torch.where(
                g_origin,
                positions,
                -1,
            )

            last_reset = reset_positions.cummax(dim=1).values

            angles = torch.cat([
                torch.zeros_like(angle_pi[:, :1]),
                angle_pi,
            ], dim=1)

            reference_angle = angles.gather(
                dim=1,
                index=last_reset + 1,
            )

            if self.use_h2:
                angles_h2 = torch.cat([
                    torch.zeros_like(angle_pi_h2[:, :1]),
                    angle_pi_h2,
                ], dim=1)

                reference_angle_h2 = angles_h2.gather(
                    dim=1,
                    index=last_reset + 1,
                )

        working_angle = angle_pi - reference_angle

        if self.use_h2:
            working_angle_h2 = angle_pi_h2 - reference_angle_h2

        # Only H1 acts on N.
        cos_theta = torch.cos(working_angle)
        sin_theta = torch.sin(working_angle)

        vx = movement_magnitude[..., 0]
        vy = movement_magnitude[..., 1]

        allo_velocity = torch.stack(
            [
                vx * cos_theta - vy * sin_theta,
                vx * sin_theta + vy * cos_theta,
            ],
            dim=-1,
        )

        allocentric_velocity = self.out_proj(allo_velocity)

        value_rotation = None

        if self.config.rotate_values:
            if self.use_h2:
                # H1 -> first half, H2 -> second half
                half = self.n_blocks // 2

                value_angle = torch.cat([
                    working_angle.unsqueeze(-1).expand(
                        -1, -1, -1, half
                    ),
                    working_angle_h2.unsqueeze(-1).expand(
                        -1, -1, -1, half
                    ),
                ], dim=-1)

            else:
                # Original formulation: H1 acts on ALL values.
                value_angle = working_angle.unsqueeze(-1).expand(
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

        if self.use_h2:
            velocities_norms = velocities_norms + 4.0 * (
                1.0 - torch.cos(angular_velocity_h2)
            )

        return allocentric_velocity, {
            "activity_norm_lie_algebra": velocities_norms,
            "value_rotation": value_rotation,
            "allo_velocity": allo_velocity,
            "head_angle": angular_velocity,
            "reference_angle": reference_angle,
            "working_angle": working_angle,
            "origin_gate": origin_gate,
            "head_angle_h2": (
                angular_velocity_h2 if self.use_h2 else None
            ),
            "reference_angle_h2": (
                reference_angle_h2 if self.use_h2 else None
            ),
            "working_angle_h2": (
                working_angle_h2 if self.use_h2 else None
            ),
        }
