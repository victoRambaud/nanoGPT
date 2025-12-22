import torch
import torch.nn as nn

from typing import Optional, Tuple, Any

from models.transformer_utils import TransformerConfig, Block, LayerNorm
from models.transformer import CausalSelfAttention
from models.normed_em_transformer import nEMBlock


class EMWMTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList(
            [
                (
                    nEMBlock(config)
                    if (l > 0 and l % 4 == 0)
                    else Block(
                        config=config,
                        attention_module=CausalSelfAttention,
                        cope_module=None,
                        layer_index=(config.n_layer - 1 - l),
                    )
                )
                for l in range(config.n_layer)
            ]
        )
        self.out_norm = LayerNorm(config.n_embd, bias=config.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Any]:

        out_dict_main = dict()
        for i, block in enumerate(self.blocks):
            x, out_dict = block(x)
            # for k, v in out_dict.items():
            #     out_dict_main[f"{k}_layer{str(i)}"] = v

        # if temperature is not None:
        #     out_dict_main["temperature"] = temperature
        return self.out_norm(x), out_dict_main
