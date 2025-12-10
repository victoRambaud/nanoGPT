import math
from copy import deepcopy
import time


from typing import *

import torch_parallel_scan as tps

import torch
import torch.nn as nn
from torch.nn import functional as F

from rotary_embedding_torch import RotaryEmbedding

from models.transformer_utils import (
    CoPE,
    TransformerConfig,
    LayerNorm,
    Block,
    RotationModule,
    naive_cum_sum
)
from typing import Optional
        

class CausalSelfAttention(nn.Module):

    def __init__(self, config: TransformerConfig, cope_module: Optional[CoPE] = None, layer_index: Optional[float] = None):
        super().__init__()
        self.config = config

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.temperature = config.temperature


        # if we share a projection for cope key and query we need to have
        self.cope_shared_key_query = config.cope_shared_key_query
        if self.cope_shared_key_query:
            config.sep_key = True

        self.cope_key = (
            nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            if config.cope and config.sep_key
            else None
        )
        self.cope_queries = (
            nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            if config.cope and config.sep_query and not config.cope_shared_key_query
            else None
        )
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head

        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        if cope_module is not None:
            print("Single CoPE module for whole Transformer")
            self.cope = cope_module
        else:
            self.cope = (
                CoPE(
                    npos_max=config.cope_npos_max,
                    head_dim=self.head_dim,
                    broadcast_heads=config.cope_broadcast_heads,
                )
                if config.cope
                else None
            )

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and self.cope is None
            and not config.softmax_log_norm
        )
        # self.flash = False

        self.rotary_emb = (
            RotaryEmbedding(dim=self.head_dim // 2, theta=config.block_max_init) if config.rope else None
        )
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
        if config.woorking_memory:
            self.rotation_module = RotationModule(config, layer_index=layer_index)

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None, temperature: Optional[float]=None):
        ts = time.time()
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)

        out_dict = dict()
        if self.config.woorking_memory:
            thetaS, _, th, rot_dict = self.rotation_module(x)
            q, k = self.rotation_module.rotate_qk(thetaS, q, k)
            # q, k, theta, th, (tt, ttt), rot_dict = self.rotate_qk(x, q, k)
            out_dict["theta"] = th
            for ke, va in rot_dict.items():
                out_dict[ke] = va

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )

        else:
            # manual implementation of attention
            L, S = q.size(-2), k.size(-2)

            scale_factor = 1 / math.sqrt(q.size(-1))
            attn_weight = q @ k.transpose(-2, -1) * scale_factor
            
            if self.config.use_padding:
                temp_mask = torch.ones(
                    L, S, dtype=torch.bool, device=attn_weight.device
                ).tril(diagonal=0)
                padding_mask = padding_mask.to(torch.bool)
                valid_q = padding_mask.unsqueeze(1).unsqueeze(-1)   # [B, 1, L, 1]
                valid_k = padding_mask.unsqueeze(1).unsqueeze(2)    # [B, 1, 1, S]
                combined_mask = temp_mask.unsqueeze(0).unsqueeze(0) & valid_q & valid_k  # [B, 1, L, S]
                attn_bias = torch.zeros_like(attn_weight)
                attn_bias.masked_fill_(~combined_mask, float("-inf"))
            else:
                temp_mask = torch.ones(
                    L, S, dtype=torch.bool, device=attn_weight.device
                ).tril(diagonal=0)
                attn_bias = torch.zeros_like(attn_weight)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

            attn_weight += attn_bias
            if self.cope is not None:
                attn_weight = attn_weight + self.cope(q, attn_weight)

            if not self.training:
                out_dict["attn_weight_x"] = attn_weight
            temperature = temperature if temperature is not None else self.temperature
            attn_weight = torch.softmax(
                attn_weight / temperature,
                dim=-1,
            )
            y = attn_weight @ v

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        out_dict["attn_time"] = time.time() - ts
        out_dict["attn_weight"] = attn_weight

        out_dict["att_x_norm"] = attn_weight.abs().mean(dim=(-2, -1)).mean().item()
        out_dict["q_norm"] = q.norm(dim=-1).mean().item()
        out_dict["k_norm"] = k.norm(dim=-1).mean().item()
        return y, out_dict


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        if config.single_cope:
            cope_module = CoPE(
                npos_max=config.cope_npos_max, head_dim=config.n_embd // config.n_head
            )
        else:
            cope_module = None

        if config.attention_type == "normal":
            # attention_module = SSMAttention
            attention_module = CausalSelfAttention
        # elif config.attention_type == "ssm":
        #     attention_module = SSMAttention
        # elif config.attention_type == "ssm2":
        #     attention_module = SSMAttention2
        else:
            raise NotImplementedError

        self.blocks = nn.ModuleList(
            [
                Block(
                    config=config,
                    attention_module=attention_module,
                    cope_module=cope_module,
                    layer_index=(config.n_layer-1-l)
                )
                for l in range(config.n_layer)
            ]
        )
        self.out_norm = LayerNorm(config.n_embd, bias=config.bias)
        
        # for name, module in self.named_modules():
        #     if isinstance(module, nn.Linear):
        #         nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #         if module.bias is not None:
        #             nn.init.zeros_(module.bias)
        #     elif isinstance(module, nn.Embedding):
        #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith("c_proj.weight"):
        #         torch.nn.init.normal_(
        #             p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
        #         )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, temperature: Optional[float]=None) -> Tuple[torch.Tensor, Any]:
        
        out_dict_main = dict()
        for i, block in enumerate(self.blocks):
            x, out_dict = block(x, padding_mask, temperature=temperature)
            for k, v in out_dict.items():
                out_dict_main[f"{k}_layer{str(i)}"] = v
        
        if temperature is not None:
            out_dict_main["temperature"] = temperature
        return self.out_norm(x), out_dict_main
