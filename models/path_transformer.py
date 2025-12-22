import torch
import math
import torch.nn.functional as F

from torch import nn
from typing import *
from dataclasses import dataclass

from models.transformer_utils import (
    init_two_linear_for_gain,
    TransformerConfig,
    Block,
    LayerNorm,
    RotationModule,
    CoPE,
    ExponentialCoPE,
    init_rotation_matrix,
)


@dataclass
class PathTransformerConfig(TransformerConfig):
    g_act_fn: str = "relu"
    # velocity_bottleneck: int = 2  # for velocity intrinsic dimension
    merge: str = "mul"
    share_velocity: bool = True
    sensory_attention: bool = True  # to test if we only compute attention on g
    softmax_log_norm: bool = True
    g_init: str = "randn"
    g_scale: float = 0.25

    commute: bool = True

    def __post_init__(self):
        TransformerConfig.__post_init__(self)


class PathCausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig, **kwargs):
        super().__init__()
        self.config = config

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.temperature = config.temperature

        # output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head

        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.dropout = config.dropout

        self.merge = config.merge

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

    def forward(self, x: torch.Tensor, g: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        B, L, D = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        k = k.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # q_g, k_g = self.qk_g(g).split(self.n_embd, dim=2)
        q_g = g
        k_g = g
        k_g = k_g.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q_g = q_g.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        # v = v + q_g
        q_cat = torch.concat([q, q_g], dim=1)  # (B, 2*nh, T, hs)
        k_cat = torch.concat([k, k_g], dim=1)  # (B, 2*nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        L, S = q_cat.size(-2), k_cat.size(-2)

        scale_factor = (
            1 / math.sqrt(q.size(-1))
            if not self.config.sensory_attention
            else 1 / q.size(-1)
        )
        attn_weight = q_cat @ k_cat.transpose(-2, -1) * scale_factor
        attn_weight_x, attn_weight_g = attn_weight.split(self.n_head, dim=1)

        temp_mask = torch.ones(
            L, S, dtype=torch.bool, device=attn_weight_x.device
        ).tril(diagonal=0)
        attn_bias = torch.zeros_like(attn_weight_x)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if self.config.sensory_attention:
            attn_weight = (
                attn_weight_x + attn_weight_g
                if self.merge == "add"
                else attn_weight_x * attn_weight_g
            )  # (b, n_h, l, h, h)
        else:
            attn_weight = 1e-6 * attn_weight_x + attn_weight_g

            attn_weight += attn_bias

        log_norm = (
            torch.log(torch.tensor(L).float()) if self.config.softmax_log_norm else 1.0
        )

        attn_weight = torch.softmax(
            # attn_weight / self.temperature,
            attn_weight / (log_norm * self.temperature),
            dim=-1,
        )
        
        # attn_weight = torch.softmax(attn_weight / self.temperature, dim=-1)
        # with torch.no_grad():
        #     attn_weight_g += attn_bias
        #     attn_weight_g = torch.softmax(
        #         # attn_weight / self.temperature,
        #         attn_weight_g / (log_norm * self.temperature),
        #         dim=-1,
        #     )
        #     attn_weight_x += attn_bias
        #     attn_weight_x = torch.softmax(
        #         # attn_weight / self.temperature,
        #         attn_weight_x / (log_norm * self.temperature),
        #         dim=-1,
        #     )
        y = attn_weight @ v

        y = (
            y.transpose(1, 2).contiguous().view(B, L, D)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.out_proj(y))

        out_dict = dict()
        # if not self.training:
        out_dict["attn_weight"] = attn_weight
        out_dict["attn_weight_x"] = attn_weight_x
        out_dict["attn_weight_g"] = attn_weight_g
        out_dict["att_x_norm"] = attn_weight_x.abs().mean(dim=(-2, -1)).mean().item()
        out_dict["att_g_norm"] = attn_weight_g.abs().mean(dim=(-2, -1)).mean().item()

        out_dict["g_norm"] = g.norm(dim=-1).mean().item()
        out_dict["q_norm"] = q.norm(dim=-1).mean().item()
        out_dict["k_norm"] = k.norm(dim=-1).mean().item()

        return y, out_dict


class PathIntegrationModule(nn.Module):
    def __init__(self, config: TransformerConfig, layer_index: Optional[int] = None):
        super().__init__()
        self.config = config

        self.rotation_module = RotationModule(config, layer_index=layer_index)

        # since we have a shared rotation matrix we need a shared init position
        # init_g = (
        #     torch.ones(config.head_dim)
        #     if config.g_init == "ones"
        #     else torch.randn(config.head_dim)
        # )
        # self.init_g = nn.Parameter(init_g * config.g_scale)
        self.init_g = nn.Parameter(torch.randn(config.head_dim) * 0.25)
        self.g_act_fn = nn.ReLU() if config.g_act_fn == "relu" else nn.Identity()

    def init_positions(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # b, l, nh, nb = theta.shape
        b, l, _ = x.shape
        # for init we start at initial position then do successive rotations
        # cumsum allows to do this in parallel

        g = (
            self.g_act_fn(self.init_g)
            .view(1, 1, 1, -1)
            .repeat(b, l, self.config.n_head, 1)
        )  # b, l, nh, h
        return g

    def path_integration(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # TODO if g is not None, we should do a full SSM step to integrate both g_t^l-1 and g_t-1^l
        # i.e. past step at current layer, or current step at previous layer
        # h_t^l = A_t h_t-1^l + B_t h_t^l-1
        b, l, nh, _ = g.shape

        thetaS, theta, th, rot_dict = self.rotation_module(x)
        # g = g.view(b, l, nh, self.config.n_diag_blocks, self.config.diag_block_size)
        # Path integration
        # g = torch.einsum("blhnij,blhnj->blhni", thetaS, g).view(b, l, nh, -1)
        g, _ = self.rotation_module.rotate_qk(thetaS, g)
        return self.g_act_fn(g), theta, th, rot_dict

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
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
        g = self.init_positions(x)

        g, theta, th, rot_dict = self.path_integration(x, g)
        return g.view(b, l, -1), theta, th, rot_dict


class EMBlock(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        # path_integration_module: PathIntegrationModule,
        cope_module: Optional[CoPE] = None,
        layer_index: Optional[int] = None,
    ):
        super().__init__()
        # self.path_integration_module = path_integration_module
        self.path_integration_module = PathIntegrationModule(
            config, layer_index=layer_index
        )

        self.transformer_block = Block(
            config, attention_module=PathCausalSelfAttention, cope_module=cope_module
        )  # block for x refinement once g has been path integrated

    def forward(
        self, x: torch.Tensor, g: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # TODO might need a norm layer should it need a specific attention ?
        # We could have it in the attention block since we already compute its attention
        # path integrate
        g, theta, th, rot_dict = self.path_integration_module(x, g)
        # g_pi, theta, th, rot_dict = self.path_integration_module(x, g)
        # if g is not None:
        #     g = g + g_pi
        # else:
        #     g = g_pi

        # update x like in a normal transformer
        x, out_dict = self.transformer_block(x, g)
        for k, v in rot_dict.items():
            out_dict[k] = v
        # x = g_pi
        # out_dict = dict()
        if not self.training:
            out_dict["g"] = g
        out_dict["theta"] = th
        return x, out_dict


class EMTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        cope_module = None

        # self.path_integration_module = PathIntegrationModule(config)

        self.blocks = nn.ModuleList(
            [
                EMBlock(
                    config,
                    # path_integration_module=self.path_integration_module,
                    cope_module=cope_module,
                    layer_index=(config.n_layer - 1 - l),
                )
                for l in range(config.n_layer)
            ]
        )
        self.out_norm = LayerNorm(config.n_embd, bias=config.bias)
        # init all weights
        # self.apply(self._init_weights)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "theta_embedd" not in name:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                print("name", name)
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("qkv_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, temperature: Optional[float]=None) -> Tuple[torch.Tensor]:
        g = None
        for block in self.blocks:
            x, out_dict = block(x, g)

        return x, out_dict
