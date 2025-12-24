import math

import torch
import torch.nn as nn

from typing import Optional, Tuple

from models.transformer_utils import TransformerConfig, RotationModule


def just_norm(x: torch.Tensor) -> torch.Tensor:
    res = x / x.norm(p=2, dim=-1, keepdim=True)
    return res


class PathIntegrationModule(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.rotation_module = (
            RotationModule(config)
        )

        # since we have a shared rotation matrix we need a shared init position
        # init_g = (
        #     torch.ones(config.head_dim)
        #     if config.g_init == "ones"
        #     else torch.randn(config.head_dim)
        # )
        # self.init_g = nn.Parameter(init_g * config.g_scale)
        self.init_q = nn.Parameter(torch.randn(config.head_dim))
        self.init_k = nn.Parameter(torch.randn(config.head_dim))
        self.init_v = nn.Parameter(torch.randn(config.head_dim), requires_grad=False)

        self.sqk_init_value = 1.0
        self.sqk_init_scaling = config.base_scale_ngpt
        self.sqk = nn.Parameter(self.sqk_init_scaling * torch.ones(self.config.n_embd))
        # self.g_act_fn = nn.ReLU() if config.g_act_fn == "relu" else nn.Identity()

    def init_positions(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # b, l, nh, nb = theta.shape
        b, l, _ = x.shape
        # for init we start at initial position then do successive rotations
        # cumsum allows to do this in parallel
        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, 1, self.config.n_head, self.config.n_embd // self.config.n_head
        )
        q = (
            just_norm(self.init_q)
            .view(1, 1, 1, -1)
            .repeat(b, l, self.config.n_head, 1)
        ) * sqk # b, l, nh, h
        k = (
            just_norm(self.init_k)
            .view(1, 1, 1, -1)
            .repeat(b, l, self.config.n_head, 1)
        ) * sqk  # b, l, nh, h
        # v = (
        #     just_norm(self.init_v)
        #     .view(1, 1, 1, -1)
        #     .repeat(b, l, self.config.n_head, 1)
        # ) * sqk  # b, l, nh, h
        return q, k

    def path_integration(
        self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        # TODO if g is not None, we should do a full SSM step to integrate both g_t^l-1 and g_t-1^l
        # i.e. past step at current layer, or current step at previous layer
        # h_t^l = A_t h_t-1^l + B_t h_t^l-1
        b, l, nh, _ = q.shape

        thetaS, theta, th, rot_dict = self.rotation_module(x)
        q, k = self.rotation_module.rotate_qk(thetaS, q, k)

        # Path integration
        # v = torch.einsum("blhnij,blhnj->blhni", thetaS, v).view(b, l, nh, -1)
        return q, k, theta, th, rot_dict

    def forward(
        self,
        x: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l = x.shape[:2]

        # v should be of shape b, l, nh, dt_rank
        # then, for each rotation matrix A_i, k_i of shape dt_rank, A*_i = exp(<v, k_i>A_i) = exp(v_i A_i)
        # but this is exactly like doing v = G o L, L -> (b, l, nh, dt_rank), G -> (b, l, nh, nb)
        # v_init = self.v_embedd(x).view(b, l, self.config.n_head, -1)  # b, l, nh, nb

        # if g is None:
        #     g = self.init_positions(x)
        # else:
        #     # v = v_init
        #     g = g.view(b, l, self.config.n_head, -1)
        q, k = self.init_positions(x)

        q, k, theta, th, rot_dict = self.path_integration(x, q, k)
        return q.view(b, l, -1), k.view(b, l, -1), theta, th, rot_dict


class nEMBlock(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.temperature = config.temperature

        # self.rotation_module = RotationModule(config=config)
        self.path_integation_module = PathIntegrationModule(config)

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

    def forward(self, x: torch.Tensor):
        B, L, D = x.size()

        ######################### START ATTENTION #########################

        hin = x

        q, k, v = self.qkv_proj(hin).split(self.n_embd, dim=2)
        k = k.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)

        q_p, k_p, theta, th, rot_dict = self.path_integation_module(hin)
        k_p = k_p.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q_p = q_p.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        # v_p = v_p.view(B, L, self.n_head, D // self.n_head).transpose(
        #     1, 2
        # )  # (B, nh, T, hs)
        
        # v = torch.einsum('bnlh, bnlk -> bnlhk', v, v_p).flatten(-2)

        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, self.config.n_head, 1, self.config.n_embd // self.config.n_head
        )
        q = sqk * just_norm(q)
        k = sqk * just_norm(k)

        q_cat = torch.concat([q, q_p], dim=1)  # (B, 2*nh, T, hs)
        k_cat = torch.concat([k, k_p], dim=1)

        L, S = q_cat.size(-2), k_cat.size(-2)

        scale_factor = 1 / q.size(-1) if self.config.inv_scale_attn else q.size(-1)
        attn_weight = q_cat @ k_cat.transpose(-2, -1) * scale_factor
        attn_weight_x, attn_weight_g = attn_weight.split(self.config.n_head, dim=1)

        temp_mask = torch.ones(
            L, S, dtype=torch.bool, device=attn_weight_x.device
        ).tril(diagonal=0)
        attn_bias = torch.zeros_like(attn_weight_x)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        # if self.config.sensory_attention:
        #     attn_weight = (
        #         attn_weight_x + attn_weight_g
        #         if self.config.merge == "add"
        #         else attn_weight_x * attn_weight_g
        #     )  # (b, n_h, l, h, h)
        # else:
        attn_weight = attn_weight_x * attn_weight_g

        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight / self.temperature, dim=-1)

        y = attn_weight @ v
        # y = y.view(B, self.config.n_head, L, self.config.head_dim, self.config.head_dim).sum(dim=-1)
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
        out_dict["theta"] = th
        return x, out_dict


class nEMTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([nEMBlock(config) for _ in range(config.n_layer)])

    def forward(self, x: torch.Tensor, temperature: Optional[float] = None):
        for block in self.blocks:
            x, out_dict = block(x)
        
        return x, out_dict