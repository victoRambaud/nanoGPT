import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from typing import Optional, Tuple, Dict


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
    use_padding: bool = False
    attention_type: str = "normal"  # "normal" or "ssm"
    base_scale_ngpt: float = 1.0 / (1024.0 ** 0.5)
    position_ssm: bool = (
        False  # used only in SSM attention if we want to also add positional embeddings
    )
    block_size: int = 1024
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
    sensory_attention: bool = False

    dt_rank: Optional[int] = None
    block_max_init: float = 1.0
    block_layer_scaling_ratio: float = 0.    # 
    block_min_init: Optional[int] = 0.01  # for landscale 1, we don't want
    approx_method: str = "taylor"
    n_approx_steps: int = -1  # set to 0 if torch exp is to be used
    block_share_rotation: bool = False  # use the same rotation matrices per head
    commute: bool = True
    tanh_alpha: float = 1.0

    freq_init_uniform: bool = False

    share_velocity: bool = True
    g_init: str = "randn"
    g_scale: float = 0.25

    softmax_log_norm: bool = False

    def __post_init__(self):
        if self.n_approx_steps == -1:
            self.diag_block_size = 2

        if self.dt_rank is None:
            self.dt_rank = self.diag_block_size

        self.n_head = self.n_embd // self.head_dim
        self.n_diag_blocks = self.head_dim // self.diag_block_size
        
        if self.freq_init_uniform:
            assert self.block_max_init > self.block_min_init
            self.step_block = (
                -(self.block_max_init - self.block_min_init) / self.n_diag_blocks
            )


def log_normal_arange(k, max_freq: float, device="cpu") -> torch.Tensor:
    """
    Deterministic log-skewed arange of k frequencies between 0 and 2π,
    denser near 2π.
    """
    # Uniform grid in [0,1]
    u = torch.linspace(1/k, 1+1/k, steps=k, device=device)

    # Map through log transform to skew density
    # Add epsilon to avoid log(0)
    eps = 1e-9
    skewed = torch.log(u + eps)
    # print(skewed)
    # Normalize to [0, 2π]
    skewed = (skewed - skewed.min()) / (skewed.max() - skewed.min())
    # print(skewed)
    freqs = max_freq * skewed

    # Flip so that density is higher near 2π
    # freqs = max_freq - freqs

    return freqs


def init_rotation_matrix(config: TransformerConfig, layer_index: Optional[int] = None) -> torch.Tensor:
    # TODO change init for higher dimension
    # S = torch.randn(config.n_diag_blocks, config.diag_block_size, config.diag_block_size, dtype=torch.float32)
    S = torch.zeros(
        config.n_diag_blocks,
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
        if config.block_layer_scaling_ratio > 0:
            s = 2**(1/config.block_layer_scaling_ratio)
            base_freq = config.base_freq / (s**layer_index)
        else:
            base_freq = config.base_freq
        # freqs = config.block_max_init * ((base_freq) ** (-(torch.arange(0, config.n_diag_blocks)/config.n_diag_blocks)**config.freq_init_alpha))
        freqs = config.block_max_init * ((base_freq) ** (-(torch.arange(1, 1+config.n_diag_blocks)/config.n_diag_blocks)**config.freq_init_alpha))
    # freqs = log_normal_arange(k=config.n_diag_blocks, max_freq=config.block_max_init)
    S = S * freqs.unsqueeze(-1).unsqueeze(-1)
    return S, freqs


def init_two_linear_for_gain(L: int, lin1: nn.Linear, lin2: nn.Linear):
    in_features = lin1.in_features
    hidden = lin1.out_features
    assert lin2.in_features == hidden, "Layers must connect"

    # symmetric split: g1 = g2 = sqrt(L)
    s1 = (L / in_features) ** 0.5
    s2 = (L / hidden) ** 0.5

    with torch.no_grad():
        nn.init.normal_(lin1.weight, mean=0.0, std=s1)
        nn.init.normal_(lin2.weight, mean=0.0, std=s2)
        if lin1.bias is not None:
            lin1.bias.zero_()
        if lin2.bias is not None:
            lin2.bias.zero_()


def naive_cum_sum(a: torch.Tensor) -> torch.Tensor:
    """
    a: shape (b, l, d)
    returns M: shape (b, l, l, d)
      M[:, i, j, :] = sum_{k=i+1..j} a[:, k, :] if j > i
                     = 0 otherwise
    """
    l = a.shape[1]
    # prefix sums along l
    S = torch.cumsum(a, dim=1)  # [b, l, d]

    # difference S[j] - S[i] using broadcasting
    # expand to [b, l, l, d]
    Sj = S.unsqueeze(1)  # [b, 1, l, d]
    Si = S.unsqueeze(2)  # [b, l, 1, d]
    M = Si - Sj

    # Mask out j <= i
    mask = torch.triu(torch.ones(l, l, device=a.device, dtype=torch.bool), diagonal=1).T
    M = M * mask.view(1, l, l, *(1 for _ in range(len(M.shape[3:]))))

    return M


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
        print(cope[0, 0])
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


class Block(nn.Module):

    def __init__(
        self,
        config: TransformerConfig,
        attention_module: nn.Module,
        cope_module: Optional[CoPE] = None,
        layer_index: Optional[float] = None,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = attention_module(config, cope_module=cope_module, layer_index=layer_index)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, *path_integration_args, temperature: Optional[float]=None
    ) -> Tuple[torch.Tensor, Dict]:
        x_att, out_dict = self.attn(self.ln_1(x), *path_integration_args, temperature=temperature)
        x = x + x_att
        x = x + self.mlp(self.ln_2(x))
        return x, out_dict


def pade_coeffs_sym(
    k: int, device: torch.device = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Padé approximate for k = m
    https://www.cis.upenn.edu/~cis6100/higham_matrix_exponential_siam_2004.pdf

    Args:
        k (int): Max coef

    Returns:
        _type_: P_kk and Q_kk coefs
    """
    m = k
    # Initialize P_pq(x) and Q_pq(x) as zero tensors
    P = [
        math.factorial(k + m - j) / (math.factorial(j) * math.factorial(k - j))
        for j in range(k)
    ]
    Q = [
        (-1) ** (j)
        * (math.factorial(k + m - j))
        / (math.factorial(m - j) * math.factorial(j))
        for j in range(m)
    ]

    return torch.tensor(P, device=device), torch.tensor(Q, device=device)


class LearnableFreqs(nn.Module):
    def __init__(self, freqs: torch.Tensor, ema_decay=0.98):
        super().__init__()
        
        self.freqs = nn.Parameter(freqs.log())
        self.freqs_ema = nn.Parameter(freqs.log(), requires_grad=False)
        self.ema_decay = ema_decay

    def forward(self, use_ema=True)->torch.Tensor:

        log_f = (1-self.ema_decay) * self.freqs + (self.ema_decay * self.freqs_ema)
        with torch.no_grad():
            self.freqs_ema = log_f
        f = torch.exp(log_f)
        return f
    

class LearnableFreqs2(nn.Module):
    def __init__(self, freqs: torch.Tensor, ema_decay=0.98):
        super().__init__()
        
        self.freqs = nn.Parameter(freqs)
        self.freqs_ema = nn.Parameter(freqs, requires_grad=False)
        self.ema_decay = ema_decay

    def forward(self, use_ema=True)->torch.Tensor:

        f = (1-self.ema_decay) * self.freqs + (self.ema_decay * self.freqs_ema)
        with torch.no_grad():
            self.freqs_ema.values = f
        f = torch.sqrt(f**2)
        return f


class RotationModule(nn.Module):
    def __init__(self, config: TransformerConfig, layer_index: Optional[int] = None):
        super().__init__()

        self.config = config

        self.theta_embedd = nn.Sequential(
            nn.Linear(config.n_embd, config.dt_rank),
            nn.Linear(config.dt_rank, config.n_head * config.n_diag_blocks, bias=False),

        )
        self.theta_act = DyT(
            dim=config.n_diag_blocks,
            alpha=config.tanh_alpha,
            requires_grad=True,
        ) if config.tanh_alpha > 0 else nn.Identity()
        # init_two_linear_for_gain(0.15, self.theta_embedd[0], self.theta_embedd[1])
        S, freqs = init_rotation_matrix(config, layer_index=layer_index)
        if config.n_approx_steps >= 0:
            self.S = nn.Parameter(S)

        else:
            # (b, l, nh, nb, 1)
            self.freqs = nn.Parameter(
                freqs.squeeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            # self.freq_mul = nn.Parameter(config.block_max_init*torch.ones(1))
            # self.freqs = LearnableFreqs(freqs.squeeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0))

        self.matrix_powers = None
        self.approx_coeffs = None

    def compute_mat_expension_coeffs(self):
        device = self.S.device
        if self.config.approx_method == "taylor":
            power = torch.arange(1, self.config.n_approx_steps, device=device)
            factorials = torch.cumprod(power, dim=0)
            factorials = torch.cat([torch.ones(1, device=device), factorials])
            self.approx_coeffs = factorials

        elif self.config.approx_method == "pade":
            self.approx_coeffs = pade_coeffs_sym(self.config.n_approx_steps, device)

    def compute_matrix_powers(self):

        device = self.S.device

        # store matrix powers
        # To keep A skew symetric, we force it to be equal to A - A.T
        S = self.S - self.S.transpose(-1, -2)

        accumulation = (
            torch.eye(self.config.diag_block_size, device=device)
            .unsqueeze(0)
            .repeat(self.config.n_diag_blocks, 1, 1)
        )
        matrix_powers = torch.zeros(
            self.config.n_approx_steps, *S.size(), dtype=S.dtype, device=device
        )
        matrix_powers[0] = accumulation
        for i in range(1, self.config.n_approx_steps):
            accumulation = S @ accumulation
            matrix_powers[i] = accumulation

        self.matrix_powers = matrix_powers  # n_approx_steps, nb, b, b

    def compute_taylor_powers(self, theta: torch.Tensor) -> torch.Tensor:
        """Returns [v**0, v**1, ..., v**self.n_taylor_steps-1)] in parallel
        before combining it with A (self.matrix powers) to approximate mat_exp(v*A)

        Args:
            v (torch.Tensor): (b, l, d) velocity input

        Returns:
            torch.Tensor: powers of v up to self.n_taylor_steps-1 (b, l, d, k)
        """
        device = self.matrix_powers.device
        powers = torch.arange(
            0, self.config.n_approx_steps, device=device
        ).float()  # shape (k,)

        # 1. Compute powers of abs(x)
        abs_theta_log = torch.log(theta.abs() + 1e-6).unsqueeze(
            -2
        )  # shape (b, l, d, 1)
        abs_theta_powers = torch.exp(
            abs_theta_log * powers.view(1, 1, self.config.n_approx_steps, 1)
        )  # (b, l, d, k)

        # 2. Get sign flip mask: True where x < 0 and p is odd
        theta_neg = (theta < 0).unsqueeze(-2)  # shape (b, l, d, k)
        odd_mask = (powers % 2 == 1).view(
            1, 1, 1, self.config.n_approx_steps, 1
        )  # shape (1, 1, 1, k)
        sign = torch.where(theta_neg & odd_mask, -1.0, 1.0)  # shape (b, l, d, k)
        # Odd powers do not change the sign
        return abs_theta_powers * sign  # shape (b, n, d, k)

    def compute_scale(self, theta: torch.Tensor):
        """_summary_

        Args:
            v (torch.Tensor): (b, l, d)

        Returns:
            int: max_scale
        """
        S = self.S.clamp(0, 2 * math.pi)
        S = S - S.transpose(-1, -2)
        # S = self.S - self.S.transpose(-1, -2)
        thetaS = S.view(1, 1, 1, *S.size()) * theta.view(*theta.size(), 1, 1)
        norms = torch.linalg.norm(thetaS, dim=(-2, -1))

        # Determine the scaling factor s
        s = torch.ceil(torch.log2(norms)).int()
        s = torch.where(s > 0, s, 1)
        max_s = torch.max(s)

        return max_s

    def bind_power_matrices(
        self, theta: torch.Tensor, max_scale: torch.Tensor
    ) -> torch.Tensor:

        theta_powers_signed = self.compute_taylor_powers(theta / (2**max_scale))
        theta_powers_signed = theta_powers_signed.view(
            *theta_powers_signed.size(), 1, 1
        )
        if self.config.approx_method == "taylor":
            # compute [(v/2**scale)**k for k in range(self.n_steps)]
            # scale by factorial
            matrix_powers = self.matrix_powers / self.approx_coeffs.view(
                self.config.n_approx_steps, 1, 1, 1
            )  # (n_steps, nb, b, b)
            matrix_powers = matrix_powers.view(
                1, 1, 1, *matrix_powers.size()
            )  # (1, 1, 1, k, nh, h, h)
            mat_exp = (matrix_powers * theta_powers_signed).sum(dim=3)

        elif self.config.approx_method == "pade":
            P, Q = self.approx_coeffs
            P = P.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            Q = Q.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mat_pows = self.matrix_powers.unsqueeze(0).unsqueeze(0)
            mat_exp = (P * mat_pows).sum(dim=2) / (Q * mat_pows).sum(dim=2)

        return mat_exp

    def approximate_exp(self, theta: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            v (torch.Tensor): b, l, d

        Returns:
            torch.Tensor: (b, l, d, nh, h, h)
        """

        if self.matrix_powers is None or self.training:
            self.compute_matrix_powers()
        if self.approx_coeffs is None:
            self.compute_mat_expension_coeffs()

        # scale matrix for numerical stability when v is too big
        max_scale = self.compute_scale(theta)

        # approximate matrix
        mat_exp = self.bind_power_matrices(theta, max_scale)
        # unscale matrix
        t0 = time.time()
        for _ in range(max_scale):
            mat_exp = mat_exp @ mat_exp

        return mat_exp, {"max_scale": max_scale, "mat_pow": time.time() - t0}

    def forward_torch_exp(self, theta: torch.Tensor) -> torch.Tensor:

        # S = self.S - torch.diag_embed(torch.diagonal(self.S, dim1=-2, dim2=-1))
        S = self.S - self.S.transpose(-1, -2)
        thetaS = torch.matrix_exp(
            S.view(1, 1, 1, *S.size()) * theta.view(*theta.size(), 1, 1)
        )
        return thetaS, {}

    def forward_sins(self, theta: torch.Tensor):
        freqs = torch.sqrt(self.freqs**2)
        theta = theta * freqs#.clamp(1e-6, 2*math.pi)
        M = torch.zeros(
            *theta.size(),
            self.config.diag_block_size,
            self.config.diag_block_size,
            dtype=theta.dtype,
            device=theta.device
        )
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        M = (cos, sin)
        return M, {
            # "tanh_alpha": self.theta_embedd[-1].alpha.detach().cpu().item(),
            # "tanh_alpha": self.theta_act.alpha.detach().cpu().item(),
            # "tanh_gamma": self.theta_act.gamma.detach().cpu().item(),
            # "tanh_gamma": self.theta_act..detach().cpu().item(),
            "max_freq": freqs.detach().max().item(),
            "min_freq": freqs.detach().min().item(),
            "mean_freq": freqs.detach().mean().item(),
            "var_freq": freqs.detach().var().item(),
        }
    
    def rotate_qk(
        self, rot_matrix: torch.Tensor, q: torch.Tensor, k: torch.Tensor
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
        k = k.view(b, l, nh, self.config.n_diag_blocks, self.config.diag_block_size)

        if self.config.n_approx_steps == -1:
            cos = rot_matrix[0].unsqueeze(-1)
            sin = rot_matrix[1].unsqueeze(-1)

            def fast_rotate(x: torch.Tensor) -> torch.Tensor:
                x1 = x[..., 0::2]
                x2 = x[..., 1::2]
                x_rotated_even = x1 * cos - x2 * sin
                x_rotated_odd  = x1 * sin + x2 * cos
                x = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).flatten(-2).view(b, l, nh, -1)
                return x
            
            q = fast_rotate(q)
            k = fast_rotate(k)
        else:
            q = torch.einsum("blhnij,blhnj->blhni", rot_matrix, q).view(b, l, nh, -1)
            k = torch.einsum("blhnij,blhnj->blhni", rot_matrix, k).view(b, l, nh, -1)

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
        theta = self.theta_embedd(x).view(b, l, self.config.n_head, -1)  # b, l, nh, nb
        # theta_norm = theta.norm(p=1, dim=-1).mean().item()
        # theta_var = theta.var(dim=-1).mean().item()

        theta = self.theta_act(theta)
        # theta_min = theta.abs().min().item()
        # theta_max = theta.abs().max().item()
        thetac = theta.cumsum(dim=1)  # b, l, nh, nb
        # thetac_min = thetac.abs().min().item()
        # thetac_max = thetac.abs().max().item()

        # TODO make sure that method stays stable for large number of steps
        if self.config.n_approx_steps > 0:
            mat_exp, rot_dict = self.approximate_exp(thetac)
            rot_dict["full_rotcreation"] = time.time() - t0
        if self.config.n_approx_steps == -1:
            mat_exp, rot_dict = self.forward_sins(thetac)
        else:
            mat_exp, rot_dict = self.forward_torch_exp(thetac)
        # rot_dict["theta_norm"] = theta_norm
        # rot_dict["theta_var"] = theta_var
        # rot_dict["theta_min"] = theta_min
        # rot_dict["theta_max"] = theta_max
        # rot_dict["theta_mean"] = theta.abs().mean().item()
        # rot_dict["theta_min_cumsum"] = thetac_min
        # rot_dict["theta_max_cumsum"] = thetac_max
        # rot_dict["theta_mean_cumsum"] =  thetac.abs().mean().item()
        return mat_exp, thetac, theta, rot_dict
