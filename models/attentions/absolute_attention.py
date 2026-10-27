import torch
import math
import torch.nn.functional as F

from torch import nn
from typing import *
from torch.nn.utils.parametrizations import orthogonal

from models.transformer_utils import (
    TransformerConfig,
    Norm,
)
from models.rotation_module import RotationModule
from models.non_commute_rotation_module import NonCommutativeModule


def trunc_normal_init_(
    tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0
) -> torch.Tensor:
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
            )

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


class PathIntegrationModule(nn.Module):
    def __init__(self, config: TransformerConfig, layer_index: Optional[int] = None):
        super().__init__()
        self.config = config

        self.rotation_module = (
            RotationModule(config, layer_index=layer_index)
            if config.commute
            else NonCommutativeModule(config)
        )
        self.path_ln = Norm(config)
        if self.config.value_inv:
            self.orth = nn.Linear(config.n_embd, config.n_embd, bias=False)
            nn.init.orthogonal_(self.orth.weight)
            orthogonal(self.orth, "weight")
        # since we have a shared rotation matrix we need a shared init position
        # init_g = (
        #     torch.ones(config.head_dim)
        #     if config.g_init == "ones"
        #     else torch.randn(config.head_dim)
        # )
        # self.init_g = nn.Parameter(init_g * config.g_scale)
        # self.init_g = nn.Parameter(torch.zeros(config.head_dim))
        # nn.init.trunc_normal_(self.init_g, std=0.02)
        # self.init_g = nn.Parameter(
        #     trunc_normal_init_(torch.empty(config.head_dim), std=1/math.sqrt(config.head_dim))
        # )
        self.init_g_q = nn.Parameter(
            torch.rand(config.head_dim)
            # trunc_normal_init_(torch.empty(config.head_dim), std=1/math.sqrt(config.head_dim))
        )
        if not config.same_gqk:
            self.init_g_k = nn.Parameter(
                torch.rand(config.head_dim)
                # trunc_normal_init_(torch.empty(config.head_dim), std=1/math.sqrt(config.head_dim))
            )
        else:
            self.init_g_k = self.init_g_q

    def init_positions(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # b, l, nh, nb = theta.shape
        b, l, _ = x.shape
        # for init we start at initial position then do successive rotations
        # cumsum allows to do this in parallel

        g_q = (
            self.init_g_q
            .view(1, 1, 1, -1)
            .repeat(b, l, self.config.n_head, 1)
        )  # b, l, nh, h
        g_k = (
            self.init_g_k
            .view(1, 1, 1, -1)
            .repeat(b, l, self.config.n_head, 1)
        )
        if self.config.value_inv:
            g_q = self.orth(F.relu(g_q))
            g_k = self.orth(F.relu(g_k))
        return g_q, g_k

    def path_integration(
        self, x: torch.Tensor, g_q: torch.Tensor, g_k: torch.Tensor
    ) -> torch.Tensor:
        # TODO if g is not None, we should do a full SSM step to integrate both g_t^l-1 and g_t-1^l
        # i.e. past step at current layer, or current step at previous layer
        # h_t^l = A_t h_t-1^l + B_t h_t^l-1
        b, l, nh, _ = g_q.shape

        thetaS, theta, th, rot_dict = self.rotation_module(self.path_ln(x))
        # g = g.view(b, l, nh, self.config.n_diag_blocks, self.config.diag_block_size)
        # # Path integration
        # g = torch.einsum("blhnij,blhnj->blhni", thetaS, g).view(b, l, nh, -1)
        g_q, g_k = self.rotation_module.rotate_qk(thetaS, g_q, g_k)
        if self.config.value_inv:
            g_q = F.relu(F.linear(g_q, self.orth.weight.T))
            g_k = F.relu(F.linear(g_k, self.orth.weight.T))
        return g_q, g_k, theta, th, rot_dict

    def forward(
        self,
        x: torch.Tensor,
        g_q: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l = x.shape[:2]
        g_q, g_k = self.init_positions(x)

        g_q, g_k, theta, th, rot_dict = self.path_integration(x, g_q, g_k)
        return g_q.view(b, l, -1), g_k.view(b, l, -1), theta, th, rot_dict


class AbsoluteCausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.temperature = config.temperature

        if self.config.value_inv:
            self.orth = nn.Linear(config.n_embd, config.n_embd, bias=False)
            nn.init.orthogonal_(self.orth.weight)
            orthogonal(self.orth, "weight")

        # output projection
        if self.config.attn_outproj:
            self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        else:
            self.out_proj = nn.Identity()
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head

        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.dropout = config.dropout

        self.merge = config.merge

        self.path_integration_module = PathIntegrationModule(config)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = False
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        B, L, D = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        out_dict = dict()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        g_q, g_k, _, _, rot_dict = self.path_integration_module(x)
        k = k.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, L, self.n_head, D // self.n_head)
        if self.config.value_inv:
            v = self.orth(F.relu(v))
            out_dict["v_in_activity_norm"] = v

        if self.config.rotate_values:
            rot = self.path_integration_module.rotation_module.mat_inverse(rot_dict)
            v, _ = self.path_integration_module.rotation_module.rotate_qk(
                rot, v
            )
            v = v.transpose(1, 2)
        else:
            v = v.transpose(
                1, 2
            )  # (B, nh, T, hs)

        g_k = g_k.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        g_q = g_q.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        
        if not self.training:
            out_dict["q"] = q.detach().cpu().transpose(1, 2)
            out_dict["k"] = k.detach().cpu().transpose(1, 2)
            out_dict["q_rot"] = g_q.detach().cpu().transpose(1, 2)
            out_dict["k_rot"] = g_k.detach().cpu().transpose(1, 2)
        # v = v + q_g
        q_cat = torch.concat([q, g_q], dim=1)  # (B, 2*nh, T, hs)
        k_cat = torch.concat([k, g_k], dim=1)  # (B, 2*nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        L, S = q_cat.size(-2), k_cat.size(-2)

        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_weight = q_cat @ k_cat.transpose(-2, -1) * scale_factor
        attn_weight_x, attn_weight_g = attn_weight.split(self.n_head, dim=1)

        
        temp_mask = torch.ones(
            L, S,
            dtype=torch.bool,
            device=attn_weight.device,
        ).tril(diagonal=0)

        # [1, 1, L, S]
        combined_mask = temp_mask[None, None, :, :]

        if padding_mask is not None:
            # [B, 1, 1, S]
            valid_k = padding_mask[:, None, None, :].bool()
            combined_mask = combined_mask & valid_k

        

        y = attn_weight @ v

        # Zero out outputs at padded query positions.
        if padding_mask is not None:
            y = y.masked_fill(
                ~padding_mask[:, None, :, None].bool(), 0.0
            )

        if self.config.sensory_attention:
            attn_weight = (
                attn_weight_x + attn_weight_g
                if self.merge == "add"
                else attn_weight_x * attn_weight_g
            )  # (b, n_h, l, h, h)
        else:
            attn_weight = 1e-6 * attn_weight_x + attn_weight_g

        # Broadcasts across all attention heads,
        # including your doubled EM heads.
        attn_weight = attn_weight.masked_fill(
            ~combined_mask, float("-inf")
        )

        attn_weight = torch.softmax(
            attn_weight / self.temperature,
            # attn_weight / (log_norm * self.temperature),
            dim=-1,
        )
        y = attn_weight @ v

        if self.config.rotate_values:
            rot = rot_dict["value_rotation"]
            y, _ = self.path_integration_module.rotation_module.rotate_qk(
                # Holonomy, we must rotate by the inverse matrix
                # to go back to current frame
                rot,
                y.transpose(1, 2)
            )
            if self.config.value_inv:
                y = F.relu(F.linear(y, self.orth.weight.T))
                out_dict["v_out_activity_norm"] = y

            y = y.contiguous().view(B, L, D)
        else:
            y = (
                y.transpose(1, 2).contiguous().view(B, L, D)
            )  # re-assemble all head outputs side by side

        # y = (
        #     y.transpose(1, 2).contiguous().view(B, L, D)
        # )  # re-assemble all head outputs side by side
        out_dict["retrieved_values"] = y.detach().cpu()
        # output projection
        y = self.resid_dropout(self.out_proj(y))

        # if not self.training:
        out_dict["attn_weight"] = attn_weight
        out_dict["attn_weight_x"] = attn_weight_x
        out_dict["attn_weight_g"] = attn_weight_g
        out_dict["att_x_norm"] = attn_weight_x.abs().mean(dim=(-2, -1)).mean().item()
        out_dict["att_g_norm"] = attn_weight_g.abs().mean(dim=(-2, -1)).mean().item()

        out_dict["g_norm"] = g_q.norm(dim=-1).mean().item()
        out_dict["q_norm"] = q.norm(dim=-1).mean().item()
        out_dict["k_norm"] = k.norm(dim=-1).mean().item()

        # q = q * scale_factor
        # k = k * scale_factor
        # g_q = g_q * scale_factor
        # g_k = g_k * scale_factor
        out_dict["qg_diff"] = (
            (
                (((q.norm(dim=-1).mean(dim=-1)+1e-6)/(g_q.norm(dim=-1).mean(dim=-1)+1e-6)).log()) ** 2
                + (((k.norm(dim=-1).mean(dim=-1)+1e-6)/(g_k.norm(dim=-1).mean(dim=-1)+1e-6)).log()) ** 2
            )
            .sum(dim=1)
            .mean()
        )
        # out_dict["qg_diff"] = (
        #     (g_q.norm(dim=-1) - q.norm(dim=-1)) ** 2
        #     + (g_k.norm(dim=-1) - k.norm(dim=-1)) ** 2
        # ).mean(dim=-1)

        return y, out_dict
