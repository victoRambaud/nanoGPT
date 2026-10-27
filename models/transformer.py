import math
from copy import deepcopy
import time


from typing import *

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.attentions import CausalSelfAttention, AbsoluteCausalSelfAttention

from models.transformer_utils import (
    CoPE,
    TransformerConfig,
    Norm,
    Block,
)
from typing import Optional


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
        elif config.attention_type == "absolute":
            # attention_module = SSMAttention
            attention_module = AbsoluteCausalSelfAttention
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
        self.out_norm = Norm(config)
        
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
            # for k, v in out_dict.items():
            #     out_dict_main[f"{k}_layer{str(i)}"] = v
        
        # if temperature is not None:
        #     out_dict_main["temperature"] = temperature
        return self.out_norm(x), out_dict_main
