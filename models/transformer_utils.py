import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from typing import Optional, Tuple, Dict, List


class DyT(nn.Module):
    def __init__(self, dim: int, alpha: float = 1.0, requires_grad: bool = True):
        super().__init__()
        # self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=requires_grad)
        self.alpha = nn.Parameter(1/torch.arange(dim+1, 1, step=-1) * alpha, requires_grad=requires_grad)
        self.alpha._no_weight_decay = True
        self.tanh = nn.Tanh()
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=False)
        # self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(self.alpha * x)
        # return self.gamma * x + self.beta


@dataclass
class TransformerConfig:
    transformer_type: str = "WM"    # WM or EM
    use_padding: bool = False
    attention_type: str = "normal"  # "normal" or "ssm"
    base_scale_ngpt: float = 1.0 / (1024.0 ** 0.5)
    position_ssm: bool = (
        False  # used only in SSM attention if we want to also add positional embeddings
    )
    block_size: int = 1024
    rope_theta: int = 10000
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    # out_vocab_size: int = (
    #     -1
    # )  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    head_dim: int = 64
    # n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    rope: bool = False
    absolute_pe: bool = False
    cope: bool = False
    cope_type: str = "normal"
    cope_bind: str = "add"  # do we add or multiply the CoPE attention weights ?
    cope_npos_max: int = 1024
    single_cope: bool = False
    sep_key: bool = False
    sep_query: bool = False
    cope_shared_key_query: bool = False
    transformer_type: str = "normal"
    cope_broadcast_heads: bool = False
    path_module: bool = False
    struct_cope: bool = False
    temperature: float = 1.0
    return_head: bool = False
    same_block_init: bool = True
    base_freq: int = 1024
    freq_init_alpha: float = 1.0

    g_act_fn: str = "relu"
    diag_block_size: int = 2  # for velocity intrinsic dimension
    merge: str = "mul"
    working_memory: bool = False  # if we learn RoPE like rotations on keys and queries
    sensory_attention: bool = True

    inv_scale_attn: bool = False
    em_qk_positions: bool = True
    dt_rank: Optional[int] = None
    block_max_init: float = 1.0
    block_layer_scaling_ratio: float = 0.    # 
    block_min_init: Optional[int] = 0.01  # for landscale 1, we don't want
    approx_method: str = "taylor"
    n_approx_steps: int = -1  # set to 0 if torch exp is to be used
    block_share_rotation: bool = False  # use the same rotation matrices per head
    commute: bool = True
    tanh_alpha: float = 1.0
    follow_rank: bool = False
    init_same_head: bool = True
    shared_inner_theta: bool = True
    log_freq: bool = False
    freq_grad: bool = True

    freq_init_uniform: bool = False

    share_velocity: bool = True
    g_init: str = "randn"
    g_scale: float = 0.25

    softmax_log_norm: bool = False
    path_householder: bool = False

    # egoformer
    ego2allo: bool = False
    share_ego_encoders: bool = False
    full_ego_path: bool = False
    rotate_values: bool = False
    value_base_freq: float = 1/8
    linear_decay: float = 5.0
    attn_outproj: bool = True
    angle_scale: float = 3.14/2
    single_freq: bool = False
    zero_freqs: bool = False
    diff_norms: bool = False
    out_norm: bool = True
    use_mlp: bool = True

    def __post_init__(self):
        # if self.n_approx_steps == -1:
        #     self.diag_block_size = 2

        if self.dt_rank is None:
            self.dt_rank = self.diag_block_size

        self.n_head = self.n_embd // self.head_dim
        self.n_diag_blocks = self.head_dim // self.diag_block_size

        if self.ego2allo:
            self.shared_inner_theta = False

        if not self.ego2allo:
            self.rotate_values = False
        
        if self.freq_init_uniform:
            assert self.block_max_init > self.block_min_init
            self.step_block = (
                -(self.block_max_init - self.block_min_init) / self.n_diag_blocks
            )


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


def init_rotation_matrix(
    config: TransformerConfig,
    layer_index: Optional[int] = None,
    base_pow: int = 1,
    base_freq: Optional[int] = None,
    follow_rank: Optional[bool] = None,
    hiera: bool = False,
    config_blocks: Optional[int] = None
) -> Tuple[torch.Tensor]:
    if config_blocks is None:
        config_blocks = config.n_diag_blocks
    # TODO change init for higher dimension
    S = torch.zeros(
        config_blocks,
        config.diag_block_size,
        config.diag_block_size,
        dtype=torch.float32,
    )
    S[:, 0, -1] = 1.0  # e.g., tensor([3, 7, 0, 5, 1, 9, 8, 6, 2, 4])

    if config.freq_init_uniform:
        freqs = torch.arange(
            config.block_max_init, config.block_min_init, step=config.step_block
        )
    else:
        if base_freq is None:
            base_freq = config.base_freq if not hiera else config.base_freq_h
        if follow_rank is None:
            follow_rank = config.follow_rank

        if not config.follow_rank:
            freqs = config.block_max_init * (
                (base_freq)
                ** (
                    -(
                        (
                            torch.arange(1, 1 + config_blocks)
                            / config_blocks
                        )
                        ** config.freq_init_alpha
                    )
                )
            )
        else:
            nbs = blocks_to_repeat(config_blocks, config.dt_rank)
            freq_list = list()
            for n_diag_blocks in nbs:
                # n_diag_blocks = config.n_diag_blocks // config.dt_rank
                freqs = config.block_max_init * (
                    (base_freq)
                    ** (
                        -(
                            (torch.arange(1, 1 + n_diag_blocks) / n_diag_blocks)
                            ** config.freq_init_alpha
                        )
                    )
                )
                freq_list.append(freqs)

            freqs = torch.concat(freq_list, dim=0)

    S = S * freqs.unsqueeze(-1).unsqueeze(-1)
    return S, freqs


class CoPE(nn.Module):
    def __init__(self, npos_max: int, head_dim: int, broadcast_heads: bool = False):
        super().__init__()
        self.npos_max = npos_max
        self.broadcast_heads = broadcast_heads
        self.pos_emb = nn.parameter.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, query: torch.Tensor, attn_logits: torch.Tensor) -> torch.Tensor:
        # compute positions, but shouldn't the gates be zero on the diagonal ??
        gates = torch.sigmoid(attn_logits)

        # we mask the diagonal since the gates should be 0 on it.
        # Indeed, a token's relative position to itself is always 0
        b, h, l, l = gates.size()
        mask = torch.eye(l, device=gates.device).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(b, h, l, l)
        gates = gates * (1 - mask)

        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)

        # Do we want different distances per head ? We might but is that compatible with chunking ?
        if self.broadcast_heads:
            pos = pos[:, :1].repeat(1, h, 1, 1)

        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        cope = logits_ceil * w + logits_floor * (1 - w)

        return cope


class ExponentialCoPE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.A_log = nn.Parameter(torch.tensor(1.0))
        self.A_log._no_weight_decay = True

    def forward(self, query: torch.Tensor, attn_logits: torch.Tensor) -> torch.Tensor:
        # compute positions, but shouldn't the gates be zero on the diagonal ??
        gates = torch.sigmoid(attn_logits)

        # we mask the diagonal since the gates should be 0 on it.
        # Indeed, a token's relative position to itself is always 0
        b, h, l, l = gates.size()
        mask = torch.eye(l, device=gates.device).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(b, h, l, l)
        gates = gates * (1 - mask)

        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        A = -torch.exp(self.A_log.float())
        cope = torch.exp(A * pos)
        return cope


class MLP(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class HouseHolderAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        from fla.layers.path_attn import PaTHAttention
        from fla.layers.gated_deltaproduct import GatedDeltaProduct

        self.attn = GatedDeltaProduct(
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            head_dim=config.head_dim,
            num_householder=config.num_householder,
            use_forget_gate=False,
            use_output_gate=False,
            use_short_conv=False
        )
        # self.attn = PaTHAttention(hidden_size=config.n_embd, num_heads=config.n_head)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ):
        out, _, _ = self.attn(x)
        return out, {}


class Block(nn.Module):

    def __init__(
        self,
        config: TransformerConfig,
        attention_module: nn.Module,
        cope_module: Optional[CoPE] = None,
        layer_index: Optional[float] = None,
    ):
        super().__init__()
        self.config = config
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias) if not config.diff_norms else nn.Identity()

        if config.path_householder:
            self.attn = HouseHolderAttention(config)
        else:
            self.attn = attention_module(
                config, cope_module=cope_module, layer_index=layer_index
            )
        self.ln_2 = (
            LayerNorm(config.n_embd, bias=config.bias)
            if config.use_mlp
            else nn.Identity()
        )
        self.mlp = MLP(config) if config.use_mlp else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *path_integration_args,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict]:
        x_att, out_dict = self.attn(
            self.ln_1(x), *path_integration_args, temperature=temperature
        )
        x = x + x_att
        if self.config.use_mlp:
            x = x + self.mlp(self.ln_2(x))
        return x, out_dict


def compute_lreg(g: torch.Tensor) -> torch.Tensor:
    return (g.abs()).sum(dim=-1)
    # return (g**2).sum(dim=tuple(range(2, g.ndim)))
