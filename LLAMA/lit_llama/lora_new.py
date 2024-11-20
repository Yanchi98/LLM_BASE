# Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Dict, List

import lit_llama.model as llama

from contextlib import contextmanager
from dataclasses import dataclass


class LoRALayerFull():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class MergedLinearFull(nn.Linear, LoRALayerFull):
    def __init__(
        self, 
        # ↓ this part is for pretrained weights
        in_features: int, 
        out_features: int, 
        # ↓ the remaining part is for LoRA
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayerFull.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out

        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros(r, in_features))  # (4, 128)
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features , r))  # (256, 2)
            )
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False 
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        
        def T(w):
            return w.T if self.fan_in_fan_out else w
        
        nn.Linear.train(self, mode)
        should = self.merged if mode else not self.merged
        if self.merge_weights and should:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0),   
                    self.lora_B.data.unsqueeze(-1), 
                ).squeeze(0) 
                sign = -1 if mode else 1
                self.weight.data += sign * delta_w * self.scaling 
            self.merged = not mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)  
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)  
                after_B = F.conv1d(
                    after_A.transpose(-2, -1),  
                    self.lora_B.unsqueeze(-1),  
                ).transpose(-2, -1)  
                result += after_B * self.scaling  
            return result


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias: 
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayerFull) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    """Return state_dict with weights of LoRA's A and B matrices and with biases depending on the `bias` value.

    Args:
        model: model with LoRA layers
        bias: 
            ``"none"``: state dict will not store bias weights,
            ``"lora_only"``: state dict will store bias weights only from LoRA layers,
            ``"all"``: state dict will store all bias weights.

    Returns:
        Weights and biases of LoRA layers

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


@dataclass
class LoRAConfig:
    r: float = 0.0
    alpha: float = 1.0
    dropout: float = 0.0


class CausalSelfAttention(llama.CausalSelfAttention):
    lora_config = None

    def __init__(self, config: llama.LLaMAConfig) -> None:
        """Causal self-attention with calculating qkv matrices with a single matrix* and Low Ranking Adaptation for
        parameter-efficient fine-tuning.

        *Instead of creating multiple heads and concatenating the result (in addition to creating separate matrices for
        query, key and value for each head) we can do this in a single pass with a single weight matrix.

        Args:
            config: 
                ``"block_size"``: size of the context of the model,
                ``"vocab_size"``: number of unique tokens,
                ``"padded_vocab_size"``: padded size of the vocabulary to the nearest multiple of 64 (leads to a greater performance),
                ``"n_layer"``: number of transformer blocks (self-attention + MLP),
                ``"n_head"``: number of heads in multi-head attention mechanism,
                ``"n_embd"``: size of the embedding: vector representation of each token.
        """
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = MergedLinearFull(
            in_features=config.n_embd,
            out_features=3 * config.n_embd,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            enable_lora=[True, False, True],
            fan_in_fan_out = False,
            merge_weights=True,
            bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None


@contextmanager
def lora(r, alpha, dropout, enabled: bool = True):
    """Apply context manager under which you can instantiate the model with LoRA.

    In a nutshell the code inside this function forces to use LoRA variant of causal self-attention
    instead of the original one (without LoRA).

    Args:
        r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        enabled: enables/disables LoRA
    """
    if not enabled:
        yield
        return

    CausalSelfAttention.lora_config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)
    # when entering context manager replace link to causal self-attention class from original
    # to a variant with LoRA
    causal_self_attention = llama.CausalSelfAttention
    llama.CausalSelfAttention = CausalSelfAttention
    yield
    # when exiting context manager - restore link to original causal self-attention class
    llama.CausalSelfAttention = causal_self_attention

    CausalSelfAttention.lora_config = None
