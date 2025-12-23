import math

import torch
import torch.nn as nn

from typing import Optional, Tuple
from rotary_embedding_torch import RotaryEmbedding

from models.transformer_utils import TransformerConfig, RotationModule


def just_norm(x: torch.Tensor) -> torch.Tensor:
    res = x / x.norm(p=2, dim=-1, keepdim=True)
    return res


class nWMBlock(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.temperature = config.temperature

        self.rotation_module = RotationModule(config=config) if not config.rope else None
        self.rotary_emb = (
            RotaryEmbedding(dim=config.head_dim // 2, theta=config.rope_theta) if config.rope else None
        )

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = config.base_scale_ngpt
        self.attn_alpha = nn.Parameter(
            self.attn_alpha_init_scaling * torch.ones(self.config.n_embd)
        )

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = config.base_scale_ngpt
        self.mlp_alpha = nn.Parameter(
            self.mlp_alpha_init_scaling * torch.ones(self.config.n_embd)
        )

        self.sqk_init_value = 1.0
        self.sqk_init_scaling = config.base_scale_ngpt
        self.sqk = nn.Parameter(self.sqk_init_scaling * torch.ones(self.config.n_embd))

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.suv = nn.Parameter(
            self.suv_init_scaling * torch.ones(2 * 4 * config.n_embd)
        )

    def rotate_qk(
        self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor
        # self, theta: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate queries and keys by an angle of theta, using exponential of matrix S

        Args:
            theta (torch.Tensor): _description_
            q (torch.Tensor): _description_
            k (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        b, l, nh, _ = q.shape

        thetaS, theta, th, rot_dict = self.rotation_module(x)
        q, k = self.rotation_module.rotate_qk(thetaS, q, k)

        return q, k, theta, th, (None, None), rot_dict

    def forward(self, x: torch.Tensor, temperature: Optional[float] = None):
        B, L, D = x.size()

        ######################### START ATTENTION #########################

        hin = x

        q, k, v = self.qkv_proj(hin).split(self.n_embd, dim=2)
        k = k.view(B, L, self.n_head, D // self.n_head)
        q = q.view(B, L, self.n_head, D // self.n_head)
        v = v.view(B, L, self.n_head, D // self.n_head)

        if self.rotation_module is not None:
            q, k, theta, th, (tt, ttt), rot_dict = self.rotate_qk(x, q, k)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, self.config.n_head, 1, self.config.n_embd // self.config.n_head
        )
        q = sqk * just_norm(q)
        k = sqk * just_norm(k)

        L, S = q.size(-2), k.size(-2)

        scale_factor = math.sqrt(q.size(-1))
        # attn_weight = q @ k.transpose(-2, -1) * scale_factor

        # temp_mask = torch.ones(
        #     L, S, dtype=torch.bool, device=attn_weight.device
        # ).tril(diagonal=0)
        # attn_bias = torch.zeros_like(attn_weight)
        # attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        # attn_weight += attn_bias
        # attn_weight = torch.softmax(attn_weight / self.temperature, dim=-1)

        # y = attn_weight @ v
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0,
            is_causal=True,
            scale=scale_factor
        )
        y = y.transpose(1, 2).contiguous().view(B, L, D)

        ######################### END ATTENTION #########################

        h_att = self.att_c_proj(y)
        lr = self.attn_alpha * (
            self.attn_alpha_init_value / self.attn_alpha_init_scaling
        )
        lr = torch.abs(lr)

        A_norm = just_norm(x)  # normally, normalization is not needed
        B_norm = just_norm(h_att)

        # res = (1.0 - lr) * A_norm + lr * B_norm
        res = A_norm + lr * (B_norm - A_norm)
        x = just_norm(res)

        hin = x
        uv = self.c_fc(hin)
        suv = self.suv * (
            (self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd**0.5)
        )
        uv = suv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)

        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = just_norm(x)  # normally, normalization is not needed
        B_norm = just_norm(h_mlp)

        # res = (1.0 - lr) * A_norm + lr * B_norm
        res = A_norm + lr * (B_norm - A_norm)
        x = just_norm(res)

        out_dict = dict()
        # out_dict["theta"] = th
        # for ke, va in rot_dict.items():
        #     out_dict[ke] = va
        return x, out_dict


class nWMTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([nWMBlock(config) for _ in range(config.n_layer)])

    def forward(self, x: torch.Tensor, temperature: Optional[float] = None):
        for block in self.blocks:
            x, out_dict = block(x)
        
        return x, out_dict
