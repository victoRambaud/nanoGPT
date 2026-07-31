import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_parallel_scan as tps

from typing import Tuple, List, Optional

from models.transformer_utils import TransformerConfig


def blocks_to_repeat(num_blocks: int, dt_rank: int) -> List[int]:
    """Repeats the initialization based on rank size.
    Allows to initialize different axis in the case of dt_rank-naigation (ND-navigation)

    Args:
        num_blocks (int): _description_
        dt_rank (int): _description_

    Returns:
        List[int]: List of block size dedicated to each dimension
    """
    nbs = list()
    m = num_blocks // dt_rank
    for n in range(dt_rank):
        if n < dt_rank - 1:
            nbs.append(m)
        else:
            nbs.append(m + (num_blocks % dt_rank))
    return nbs


def init_non_commute_rotation_matrix(
    config: TransformerConfig,
    n_diag_blocks: Optional[int] = None,
    base_freq: Optional[int] = None,
) -> torch.Tensor:
    # TODO change init for higher dimension
    if n_diag_blocks is None:
        n_diag_blocks = config.n_diag_blocks
    n_matrices = config.diag_block_size * (config.diag_block_size -1) // 2
    S = torch.zeros(
        n_matrices,
        n_diag_blocks,
        config.diag_block_size,
        config.diag_block_size,
        dtype=torch.float32,
    )

    k = 0
    for i in range(1, config.diag_block_size):
        for j in range(i):
            S[k, :, j, i] = 1.
            k += 1

    # If you want to permute an arange explicitly:
    if base_freq is None:
        base_freq = config.base_freq
    if not config.follow_rank:
        
        freqs = config.block_max_init * (
            (base_freq) ** (
                -(
                    (torch.arange(1, 1 + n_diag_blocks) / n_diag_blocks)
                    ** config.freq_init_alpha
                )
            )
        )
    else:
        nbs = blocks_to_repeat(n_diag_blocks, config.dt_rank)
        freq_list = list()
        for nb in nbs:
            freqs = config.block_max_init * (
                (base_freq) ** (
                    -(
                        (torch.arange(1, 1 + nb) / nb)
                        ** config.freq_init_alpha
                    )
                )
            )
            freq_list.append(freqs)
        
        freqs = torch.concat(freq_list, dim=0)
    freqs = freqs.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return S, freqs


class FastMultiHeadFactorized(nn.Module):
    """
    Fast version: combine W_in @ W_out into single matrix per (head, generator).
    """
    def __init__(self, d_model, n_heads, n_generators=3, n_blocks=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_generators = n_generators
        self.n_blocks = n_blocks
        
        # Combined projection: [n_heads, n_generators, d_model, n_blocks]
        self.W_combined = nn.Parameter(
            torch.randn(n_heads, n_generators, d_model, n_blocks) * 0.01
        )

    def forward(self, x):
        """
        Single einsum for maximum speed.
        """
        # theta: [B, L, n_heads, n_generators, n_blocks]
        theta = torch.einsum('bld,hgdn->blhgn', x, self.W_combined)
        return theta


class NonCommutativeModule(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        # n_matrices = 1
        self.n_matrices = (config.diag_block_size * (config.diag_block_size -1)) // 2
        self.theta_embedd = FastMultiHeadFactorized(
            d_model=config.n_embd,
            n_heads=config.n_head,
            n_generators=self.n_matrices,
            n_blocks=config.n_diag_blocks,
            # rank=config.dt_rank
        )

        # init_two_linear_for_gain(1.0, self.theta_embedd[0], self.theta_embedd[1])
        S, freqs = init_non_commute_rotation_matrix(config)
        self.freqs = nn.Parameter(freqs)
        self.S = nn.Parameter(S, requires_grad=False)

    def rotate_qk(
        self,
        rot_matrix: torch.Tensor,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
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

        q = torch.einsum("blhnij,blhnj->blhni", rot_matrix, q).view(b, l, nh, -1)
        if k is not None:
            k = torch.einsum("blhnij,blhnj->blhni", rot_matrix, k).view(
                b, l, nh, -1
            )

        return q, k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = self.theta_embedd(x)
        S = (self.S - self.S.transpose(-1, -2)) * self.freqs
        S = S.view(1, 1, 1, *S.size())
        thetaS = (theta.view(*theta.size(), 1, 1) * S).sum(dim=-4)
        M = torch.matrix_exp(thetaS)
        activity_norm = theta.norm(dim=-1).sum(dim=(-2, -1)).mean()
        # cumprod along the time dimension
        M = tps.prefix_scan(M, prefix_func=torch.matmul, dim=1)
        return M, theta, theta, {"activity_norm": activity_norm}


class EgoNDimEmbedder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.config = config
        self.rank = config.dt_rank
        self.n_angles = self.rank * (self.rank - 1) // 2
        self.angular_embed = FastMultiHeadFactorized(
            d_model=config.n_embd,
            n_heads=config.n_head,
            n_generators=self.n_angles,
            n_blocks=1
        )
        # self.angular_embed = nn.Linear(
        #     config.n_embd, config.n_head * (self.n_angles), bias=False
        # )
        self.allocentric_velocity = nn.Linear(
            config.n_embd, config.n_head * (self.rank), bias=False
        )
        self.out_proj = nn.Linear(self.rank, config.n_diag_blocks, bias=False)

        # to compute inner rotations
        S, _ = init_non_commute_rotation_matrix(config, n_diag_blocks=1)
        S = S[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.inner_S = nn.Parameter(S - S.transpose(-2, -1), requires_grad=False)

        # for rotating the values
        S, value_freqs = init_non_commute_rotation_matrix(config, base_freq=config.value_base_freq)
        S = S.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.value_S = nn.Parameter(S - S.transpose(-2, -1), requires_grad=False)
        self.value_freqs = nn.Parameter(value_freqs.unsqueeze(0).unsqueeze(0).unsqueeze(0).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.size()

        # b, l, h, rank
        movement_magnitude = self.allocentric_velocity(x).view(
            b, l, self.config.n_head, -1
        )

        # b, l, h, n_angles, 1, 1
        angular_velocity = self.angular_embed(x).view(
            b, l, self.config.n_head, -1, 1, 1
        )
        # b, l, h, r, r
        heads_rotations = torch.matrix_exp((self.inner_S * angular_velocity).sum(dim=-3))
        heads_rotations_PI = tps.prefix_scan(heads_rotations, prefix_func=torch.matmul, dim=1)

        # b, l, h, r
        allocentric_velocity = torch.einsum("blhij,blhj->blhi", heads_rotations_PI, movement_magnitude)
        allocentric_velocity = self.out_proj(allocentric_velocity)

        # b, l, h, nb, r, r
        # we start by rotating by the inverse rotation to go back to 0
        value_rotation = (- angular_velocity.unsqueeze(-3) * self.value_freqs.exp() * self.value_S).sum(dim=-4)
        value_rotation = torch.matrix_exp(value_rotation)
        value_rotation = tps.prefix_scan(value_rotation, prefix_func=torch.matmul, dim=1)

        activity_norm = 2*(angular_velocity/2).sin().abs().sum(dim=(2, 3)).mean()
        
        return allocentric_velocity, {
            "activity_norm": activity_norm,
            "value_rotation": value_rotation,
        }
