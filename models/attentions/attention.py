import math
from copy import deepcopy
import time


from typing import *

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import orthogonal

from rotary_embedding_torch import RotaryEmbedding

from models.transformer_utils import (
    CoPE,
    TransformerConfig,
    Norm,
    Block,
)
from models.rotation_module import RotationModule
from models.rotors.quaternion_rotor import inv_quaternion
from typing import Optional


def make_sdpa_mask(
    padding_mask: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor]=None
) -> Tuple[torch.Tensor, bool]:
    """
    padding_mask: (B, Lq), bool
        True = valid query token

    key_padding_mask: (B, Lk), bool, optional
        True = valid key token
        None = self-attention (causal)

    returns:
        attn_mask: (B, 1, Lq, Lk), bool
            True = attention allowed
        is_causal: bool
    """
    if key_padding_mask is None:
        # Self-attention: preserve original behavior
        if padding_mask is None:
            return None, True

        B, L = padding_mask.shape
        causal = torch.tril(
            torch.ones(L, L, dtype=torch.bool, device=padding_mask.device)
        )

        key_mask = padding_mask[:, None, None, :]
        return causal[None, None, :, :] & key_mask, False

    # Cross-attention: no causal constraint
    key_mask = key_padding_mask[:, None, None, :]  # (B,1,1,Lk)

    if padding_mask is None:
        raise ValueError("Cross-attention requires a query padding mask.")

    query_mask = padding_mask[:, None, :, None]  # (B,1,Lq,1)

    return query_mask & key_mask, False


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        config: TransformerConfig,
        cope_module: Optional[CoPE] = None,
        layer_index: Optional[float] = None,
    ):
        super().__init__()
        self.config = config

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.temperature = config.temperature

        if self.config.value_inv:
            self.orth = nn.Linear(config.n_embd, config.n_embd, bias=False)
            nn.init.orthogonal_(self.orth.weight)
            orthogonal(self.orth, "weight")

        self.att_ln = nn.Identity()
        self.rot_ln = nn.Identity()
        if config.diff_norms:
            self.att_ln = Norm(config)
            if config.working_memory:
                self.rot_ln = Norm(config)

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

        if config.kv_gates:
            self.kv_gate = nn.Linear(config.n_embd, 2*config.n_head)

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
            RotaryEmbedding(dim=self.head_dim // 2, theta=config.rope_theta)
            if config.rope
            else None
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
        if config.working_memory:
            self.rotation_module = RotationModule(config, layer_index=layer_index)

    def qkv_forward(
        self,
        x: torch.Tensor,
        q_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        return q, k, v

    def rotate_qkv(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out_dict: Dict,
        q_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict, Dict]:
        thetaS, _, th, rot_dict = self.rotation_module(x)
        q, k = self.rotation_module.rotate_qk(thetaS, q, k)
        if self.config.rotate_values:
            # rotate each value by the inverse heading (which is what value_rotation returns)
            # each value is seen as if from canonical heading, all are comparable
            rot = self.rotation_module.mat_inverse(rot_dict)
            v, _ = self.rotation_module.rotate_qk(
                rot, v
            )
        # q, k, theta, th, (tt, ttt), rot_dict = self.rotate_qk(x, q, k)
        for ke, va in rot_dict.items():
            out_dict[ke] = va

        return q, k, v, out_dict, rot_dict, rot_dict
        
    def qkv_prep(
        self,
        x: torch.Tensor,
        q_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict]:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        if q_x is not None:
            _, L, _ = q_x.size()
        else:
            L = T

        rot_dict_kv, rot_dict_q = dict(), dict()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.qkv_forward(x, q_x)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, L, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)
        
        if self.config.kv_gates:
            kg, vg = self.kv_gate(x).split(self.config.n_head, -1)
            k = k * kg.sigmoid().unsqueeze(-1)
            v = v * vg.sigmoid().unsqueeze(-1)

        out_dict = dict()
        if not self.training:
            out_dict["raw_values"] = v.detach().cpu()

        if self.config.value_inv:
            v = self.orth(F.relu(v))
            out_dict["v_in_activity_norm"] = v

        if not self.training:
            out_dict["q"] = q.detach().cpu()
            out_dict["k"] = k.detach().cpu()

        if self.config.working_memory:
            q, k, v, out_dict, rot_dict_kv, rot_dict_q = self.rotate_qkv(
                x=x, q_x=q_x, q=q, k=k, v=v, out_dict=out_dict
            )

        if not self.training:
            out_dict["q_rot"] = q.detach().cpu()
            out_dict["k_rot"] = k.detach().cpu()
            flash = False

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if not self.training:
            out_dict["rotated_values"] = v.transpose(1, 2).detach().cpu()

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        return q, k, v, out_dict, rot_dict_kv, rot_dict_q

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: torch.Tensor,
        out_dict: Dict,
        key_padding_mask: Optional[torch.Tensor]=None
    ) -> Tuple[torch.Tensor, Dict]:
        scale_factor = 1 / math.sqrt(q.size(-1)) / self.temperature
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            padding_mask, is_causal = make_sdpa_mask(padding_mask, key_padding_mask=key_padding_mask)
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=padding_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=is_causal,
                scale=scale_factor,
            )

        else:
            # manual implementation of attention
            L, S = q.size(-2), k.size(-2)

            attn_weight = q @ k.transpose(-2, -1) * scale_factor

            if self.config.use_padding:
                temp_mask = torch.ones(
                    L, S, dtype=torch.bool, device=attn_weight.device
                ).tril(diagonal=0)
                padding_mask = padding_mask.to(torch.bool)
                valid_q = padding_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
                valid_k = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
                combined_mask = (
                    temp_mask.unsqueeze(0).unsqueeze(0) & valid_q & valid_k
                )  # [B, 1, L, S]
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

            temperature = temperature if temperature is not None else self.temperature
            attn_weight = torch.softmax(attn_weight, dim=-1)
            y = attn_weight @ v
            if not self.training:
                out_dict["attn_weight"] = attn_weight.detach().cpu()

        return y, out_dict

    def forward(
        self,
        x,
        padding_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ):
        B, T, C = x.size()
        q, k, v, out_dict, rot_dict_kv, rot_dict_q = self.qkv_prep(x)
        y, out_dict = self.compute_attention(
            q=q, k=k, v=v, padding_mask=padding_mask, out_dict=out_dict
        )

        if self.config.rotate_values:
            rot = rot_dict_q["value_rotation"]
            y, _ = self.rotation_module.rotate_qk(
                # Holonomy, we must rotate by the inverse matrix
                # to go back to current frame
                rot,
                y.transpose(1, 2)
            )
            if self.config.value_inv:
                y = F.relu(F.linear(y, self.orth.weight.T))
                out_dict["v_out_activity_norm"] = y

            y = y.contiguous().view(B, T, C)
            out_dict["retrieved_values"] = y.detach().cpu()
        else:

            y = (
                y.transpose(1, 2).contiguous().view(B, T, C)
            )  # re-assemble all head outputs side by side
            out_dict["retrieved_values"] = y.detach().cpu()

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if not self.training:
            out_dict["y"] = y.detach().cpu()

        return y, out_dict
