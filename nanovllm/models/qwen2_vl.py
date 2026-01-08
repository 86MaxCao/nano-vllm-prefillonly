"""Simplified implementation of the Qwen2-VL multimodal model with M-RoPE support.

This module inlines both the text backbone and the vision encoder.
Reference: transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

logger = logging.getLogger(__name__)

# Debug log file for Qwen2-VL
_DEBUG_LOG_FILE = None
_DEBUG_LOG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "log", "qwen2vl_v4.logger"
)


def _debug_log(*args, **kwargs):
    """Write debug message to both console and log file."""
    global _DEBUG_LOG_FILE
    if _DEBUG_LOG_FILE is None:
        os.makedirs(os.path.dirname(_DEBUG_LOG_FILE_PATH), exist_ok=True)
        _DEBUG_LOG_FILE = open(_DEBUG_LOG_FILE_PATH, "w", encoding="utf-8")
    # Format message like print()
    msg = " ".join(str(arg) for arg in args)
    if kwargs:
        msg += " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(msg)
    _DEBUG_LOG_FILE.write(msg + "\n")
    _DEBUG_LOG_FILE.flush()

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

# ---------------------------------------------------------------------------
# M-RoPE Utils (Ported from Transformers/vLLM)
# ---------------------------------------------------------------------------

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    Applies Multimodal Rotary Position Embedding to query and key tensors.
    
    q, k: [batch, seq_len, num_heads, head_dim] (or similar)
    cos, sin: [3, batch, seq_len, head_dim]
    mrope_section: list of 3 integers summing to head_dim, e.g. [24, 24, 24] for head_dim=72 (temporal, height, width)
    """
    # The mrope_section in config is usually for the half-dimension (rotary dimension).
    # Transformers implementation doubles it for processing.
    # q/k shape is typically [num_tokens, num_heads, head_dim] in nano-vllm context
    
    # NOTE: nano-vllm attention usually expects q, k shape as [batch_size, num_tokens, num_heads, head_dim]
    # But inside Qwen2VLTextAttention.forward below, we reshape to [batch*seq, heads, dim] or similar.
    # Let's align with the inputs passed from Attention.forward.
    
    # In Qwen2VLTextAttention below:
    # q: [batch_size, seq_len, num_heads, head_dim]
    # k: [batch_size, seq_len, num_kv_heads, head_dim]
    # cos, sin: [3, batch_size, seq_len, head_dim]
    
    # Split q/k into 3 parts along head_dim
    # mrope_section is for the rotary dimension (usually head_dim)
    # Note: Transformers impl multiplies section by 2 because it operates on cos/sin expansion?
    # Actually, mrope_section sums to head_dim (e.g. 128). 
    # Example: head_dim 128 -> mrope_section [16, 56, 56]
    
    # We need to broadcast cos/sin to match q/k heads
    # cos shape: [3, batch, seq, head_dim] -> we need to split this head_dim
    
    mrope_section = [s * 2 for s in mrope_section] # Transformers implementation does this multiplication
    
    # cos: [3, batch, seq, head_dim]
    # We split the last dim of cos/sin based on mrope_section
    
    # Helper to pick the correct cos/sin part for each section
    # i=0 (temporal), i=1 (height), i=2 (width)
    
    # q: [batch, seq, heads, head_dim] -> permute to [batch, seq, heads, head_dim] if needed
    # The input q is [batch, seq, heads, head_dim]
    
    # We need to apply rotation to each part of head_dim
    
    cos_parts = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
    sin_parts = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
    
    # Unsqueeze to broadcast over heads
    # cos_parts: [batch, seq, head_dim] -> [batch, seq, 1, head_dim]
    if unsqueeze_dim == 2:
        cos_parts = cos_parts.unsqueeze(2)
        sin_parts = sin_parts.unsqueeze(2)
    elif unsqueeze_dim == 1: # [batch, 1, seq, head_dim] - typical for [batch, heads, seq, dim] layout
        cos_parts = cos_parts.unsqueeze(1)
        sin_parts = sin_parts.unsqueeze(1)
        
    q_embed = (q * cos_parts) + (rotate_half(q) * sin_parts)
    k_embed = (k * cos_parts) + (rotate_half(k) * sin_parts)
    return q_embed, k_embed

class Qwen2VLRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size // config.num_attention_heads
        self.base = config.rope_theta
        
        # Calculate inv_freq
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # x: [batch, seq, heads, head_dim]
        # position_ids: [3, batch, seq] (temporal, height, width)
        
        # inv_freq: [head_dim/2]
        # Expand to [3, batch, seq, 1]
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        
        # position_ids: [3, batch, seq] -> [3, batch, 1, seq]
        position_ids_expanded = position_ids[:, :, None, :].float()
        
        # Output: [3, batch, head_dim/2, seq] -> transpose to [3, batch, seq, head_dim/2]
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1) # [3, batch, seq, head_dim]
            cos = emb.cos()
            sin = emb.sin()
            
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# ---------------------------------------------------------------------------
# Text backbone
# ---------------------------------------------------------------------------

class Qwen2VLTextAttention(nn.Module):

    def __init__(
        self,
        config, # Pass full config to access rope parameters
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        rms_norm_eps: float,
        qkv_bias: bool,
        head_dim: int | None,
        rope_theta: float,
        rope_scaling: tuple | None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        
        self.mrope_section = config.rope_scaling["mrope_section"] if hasattr(config, "rope_scaling") and config.rope_scaling else \
                             config.rope_parameters["mrope_section"] # Fallback

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        
        # Use Qwen2VL specific RoPE
        self.rotary_emb = Qwen2VLRotaryEmbedding(config)
        
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor, # Expected to be [3, batch, seq_len] for M-RoPE
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [batch, seq, hidden_size] or flattened
        
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Reshape for Attention
        # If flattened [total_tokens, dim], view as [batch, seq, heads, dim] based on input logic
        # Here we assume hidden_states comes in as [batch, seq, dim] because positions is [3, batch, seq]
        batch_size = positions.shape[1]
        seq_len = positions.shape[2]
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Calculate RoPE
        # positions is [3, batch, seq]
        cos, sin = self.rotary_emb(q, positions) # cos/sin: [3, batch, seq, head_dim]
        
        # Apply M-RoPE
        # unsqueeze_dim=2 because q shape is [batch, seq, heads, dim]
        # we want cos to broadcast to [batch, seq, 1, dim]
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.mrope_section, unsqueeze_dim=2)
        
        # Reshape to [total_tokens, num_heads, head_dim] format expected by Attention.forward
        # This matches the format used in qwen3_vl.py and other models
        # Flash Attention requires this format and head_dim must be a multiple of 8
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # Standard Attention
        o = self.attn(q, k, v)
        
        output = self.o_proj(o.flatten(1, -1)) # Flatten [batch, seq] -> [total_tokens]
        return output


class Qwen2VLTextMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen2VLTextDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling", None)
        # Handle dict config for rope_scaling
        
        self.self_attn = Qwen2VLTextAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=rope_scaling,
        )
        self.mlp = Qwen2VLTextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2VLTextModel(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        # 根据 tp_size 选择使用标准 Embedding 还是 VocabParallelEmbedding
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        if tp_size == 1:
            # 单 GPU：使用标准的 nn.Embedding，就像 Transformers 那样
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            # 多 GPU：使用 VocabParallelEmbedding 支持 tensor parallelism
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [Qwen2VLTextDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2VLTextForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen2VLTextModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: dict,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Vision encoder (Mostly unchanged, but ensuring compatibility)
# ---------------------------------------------------------------------------

def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1 + torch.tanh(inner))


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """Rotate half the hidden dims of the input, matching vLLM implementation."""
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    """
    Apply rotary embedding, matching vLLM implementation.
    x: (batch_size, seqlen, nheads, headdim) or similar
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def apply_rotary_pos_emb_vision(
    t: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding for vision, matching Transformers implementation."""
    # t shape: [2*b, s, head, head_dim] (concatenated q and k)
    # freqs shape: [seq_len, 2 * head_dim] (after concatenation in forward)
    # Transformers implementation:
    #   cos, sin = emb.cos(), emb.sin()  # [seq_len, 2 * head_dim]
    #   cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)  # [seq_len, 1, 2 * head_dim]
    #   cos, sin = cos[:, :head_dim], sin[:, :head_dim]  # [seq_len, 1, head_dim]
    #   q_embed = (q * cos) + (rotate_half(q) * sin)
    
    orig_dtype = t.dtype
    t_ = t.float()
    
    # freqs shape: [seq_len, 2 * head_dim]
    cos = freqs.cos()  # [seq_len, 2 * head_dim]
    sin = freqs.sin()  # [seq_len, 2 * head_dim]
    
    # t shape: [2*b, s, head, head_dim]
    # In Transformers, q/k are [seq_len, num_heads, head_dim], and after unsqueeze(-2), cos/sin are [seq_len, 1, head_dim]
    # In our case, we need to align dimensions:
    #   t: [2*b, s, head, head_dim] where s == seq_len
    #   cos/sin: [seq_len, 2 * head_dim] -> need [1, seq_len, 1, head_dim] to broadcast with [2*b, s, head, head_dim]
    
    head_dim = t.shape[-1]
    # Select first head_dim dimensions and reshape to match t's shape
    # cos/sin: [seq_len, 2 * head_dim] -> [seq_len, head_dim] -> [1, seq_len, 1, head_dim]
    cos = cos[:, :head_dim].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    sin = sin[:, :head_dim].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    
    # Apply rotary embedding: (t * cos) + (rotate_half(t) * sin)
    output = t_ * cos + rotate_half(t_) * sin
    return output.to(orig_dtype)

class Qwen2VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = getattr(config, "temporal_patch_size", 1)
        self.in_channels = config.in_channels
        self.embed_dim = getattr(config, "embed_dim", config.hidden_size)
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        stride = kernel_size
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=stride, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Transformers processor outputs flattened patch tokens: [num_patches, in_channels * temporal_patch_size * patch_size * patch_size]
        # We need to reshape to [num_patches, in_channels, temporal_patch_size, patch_size, patch_size] for Conv3d
        if inputs.dim() == 2:
            # Flattened input: [num_patches, in_channels * temporal_patch_size * patch_size * patch_size]
            # Reshape to [num_patches, in_channels, temporal_patch_size, patch_size, patch_size]
            inputs = inputs.view(
                -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
            )
        elif inputs.dim() == 4:
            # Standard 4D input: [batch, channels, height, width] -> add temporal dimension
            inputs = inputs.unsqueeze(2)
        # Now inputs is 5D: [num_patches, in_channels, temporal_patch_size, patch_size, patch_size]
        hidden_states = self.proj(inputs)
        # Output shape: [num_patches, embed_dim, 1, 1, 1] -> flatten to [num_patches, embed_dim]
        hidden_states = hidden_states.flatten(2).transpose(1, 2).squeeze(1)
        return hidden_states

class Qwen2VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class Qwen2VLVisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        embed_dim = getattr(config, "embed_dim", config.hidden_size)
        mlp_ratio = getattr(config, "mlp_ratio", 4.0)
        intermediate_size = int(embed_dim * mlp_ratio)
        self.linear_fc1 = nn.Linear(embed_dim, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, embed_dim, bias=True)
        if getattr(config, "hidden_act", "gelu") == "gelu_pytorch_tanh":
            self.act_fn = gelu_pytorch_tanh
        else:
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))

class Qwen2VLVisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        embed_dim = getattr(config, "embed_dim", config.hidden_size)
        self.dim = embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, **kwargs) -> torch.Tensor:
        # Input: [s, b, c] where s=seq_length, b=batch_size (typically 1)
        seq_length, batch_size, _ = hidden_states.shape
        
        # [s, b, c] --> [s, b, 3 * head * head_dim]
        x = self.qkv(hidden_states)
        
        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = x.chunk(3, dim=-1)
        q = q.view(seq_length, batch_size, self.num_heads, self.head_dim)
        k = k.view(seq_length, batch_size, self.num_heads, self.head_dim)
        v = v.view(seq_length, batch_size, self.num_heads, self.head_dim)
        
        # [s, b, head, head_dim] -> [b, s, head, head_dim] (matching vLLM)
        q, k, v = (rearrange(x, "s b ... -> b s ...") for x in (q, k, v))
        
        # Apply rotary position embedding (matching vLLM)
        if rotary_pos_emb is not None:
            # [2 * b, s, head, head_dim] for concatenated q and k
            qk_concat = torch.cat([q, k], dim=0)
            qk_rotated = apply_rotary_pos_emb_vision(qk_concat, rotary_pos_emb)
            q, k = torch.chunk(qk_rotated, 2, dim=0)
        
        # Use standard attention (can be upgraded to Flash Attention later)
        # Reshape for attention: [b, s, head, head_dim] -> [b, head, s, head_dim]
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # Output: [b, head, s, head_dim] -> [s, b, (head * head_dim)]
        attn_output = rearrange(attn_output, "b h s d -> s b (h d)")
        
        return self.proj(attn_output)

class Qwen2VLVisionPatchMerger(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        context_dim = getattr(config, "embed_dim", config.hidden_size)
        self.hidden_size = context_dim * (config.spatial_merge_size**2)
        self.norm = nn.LayerNorm(context_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.hidden_size, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))
        return hidden_states

class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        embed_dim = getattr(config, "embed_dim", config.hidden_size)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Qwen2VLVisionAttention(config)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = Qwen2VLVisionMLP(config)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, rotary_pos_emb, **kwargs)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class Qwen2VLVisionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Qwen2VLVisionPatchEmbed(config)
        embed_dim = getattr(config, "embed_dim", config.hidden_size)
        self.embed_dim = embed_dim
        self.rotary_pos_emb = Qwen2VLVisionRotaryEmbedding(embed_dim // config.num_heads // 2)
        self.blocks = nn.ModuleList([Qwen2VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2VLVisionPatchMerger(config=config)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        if isinstance(grid_thw, torch.Tensor): grid_thw_list = grid_thw.tolist()
        else: grid_thw_list = grid_thw
        
        max_grid_size = 0
        pos_ids = []
        for t, h, w in grid_thw_list:
            hpos = torch.arange(h, device=self.rotary_pos_emb.inv_freq.device).unsqueeze(1).expand(-1, w)
            wpos = torch.arange(w, device=self.rotary_pos_emb.inv_freq.device).unsqueeze(0).expand(h, -1)
            hpos = hpos.reshape(h//merge_size, merge_size, w//merge_size, merge_size).permute(0, 2, 1, 3).flatten()
            wpos = wpos.reshape(h//merge_size, merge_size, w//merge_size, merge_size).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
            max_grid_size = max(max_grid_size, h, w)
        
        pos_ids = torch.cat(pos_ids, dim=0)
        return self.rotary_pos_emb(max_grid_size)[pos_ids].flatten(1)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.patch_embed.proj.weight.dtype)
        hidden_states = self.patch_embed(pixel_values).reshape(-1, self.embed_dim).unsqueeze(1)
        
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        
        if isinstance(grid_thw, torch.Tensor): grid_thw_tensor = grid_thw
        else: grid_thw_tensor = torch.tensor(grid_thw, dtype=torch.int32, device=hidden_states.device)
        
        cu_seqlens = torch.repeat_interleave(grid_thw_tensor[:, 1] * grid_thw_tensor[:, 2], grid_thw_tensor[:, 0]).cumsum(0, dtype=torch.int32)
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
        
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rotary_pos_emb=emb)
        return self.merger(hidden_states).squeeze(1)

class Qwen2VisionEncoder(nn.Module):
    def __init__(self, vision_config) -> None:
        super().__init__()
        self.config = vision_config
        self.vision = Qwen2VLVisionModel(vision_config)

    def _linear_patch_embed(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        proj = self.vision.patch_embed.proj
        weight = proj.weight.view(proj.out_channels, -1)
        return F.linear(patch_tokens.to(weight.dtype), weight, proj.bias)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
        # pixel_values from processor is already flattened: [num_patches, in_channels * temporal_patch_size * patch_size * patch_size]
        # Just pass it directly to vision model, which will handle the reshaping in patch_embed
        image_embeds = self.vision(pixel_values, image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // (self.config.spatial_merge_size**2)).tolist()
        return list(torch.split(image_embeds, split_sizes))


# ---------------------------------------------------------------------------
# Multimodal wrapper (Updated with get_rope_index)
# ---------------------------------------------------------------------------

class Qwen2VLForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_config = getattr(config, "text_config", config)
        self.vision_config = getattr(config, "vision_config", None)
        if self.vision_config is None: raise ValueError("vision_config missing")

        self.visual = create_vision_model(self.vision_config)
        self.language_model = Qwen2VLTextForCausalLM(self.text_config)
        self.packed_modules_mapping = {
            "mlp.gate_proj": ("mlp.gate_up_proj", 0), "mlp.up_proj": ("mlp.gate_up_proj", 1),
            "q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"), "v_proj": ("qkv_proj", "v"),
        }

    def get_input_embeddings(self, input_ids):
        return self.language_model.model.embed_tokens(input_ids)

    def get_rope_index(self, input_ids, image_grid_thw, video_grid_thw, attention_mask=None):
        """
        Generates 3D position IDs (M-RoPE) for Qwen2-VL.
        """
        spatial_merge_size = self.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        
        device = input_ids.device
        
        # Ensure input_ids is 2D [batch, seq] for processing
        is_flattened = input_ids.dim() == 1
        if is_flattened:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
        
        position_ids = torch.zeros(3, input_ids.shape[0], input_ids.shape[1], dtype=torch.long, device=device)
        
        img_idx, vid_idx = 0, 0
        
        for i, seq_ids in enumerate(input_ids):
            valid_ids = seq_ids
            if attention_mask is not None:
                valid_ids = seq_ids[attention_mask[i] == 1]
            
            vis_start_indices = torch.argwhere(valid_ids == vision_start_token_id).squeeze(1)
            
            current_pos = 0
            llm_pos_ids_list = []
            
            if len(vis_start_indices) == 0:
                llm_pos_ids_list.append(torch.arange(len(valid_ids), device=device).view(1, -1).expand(3, -1))
            else:
                for start_idx in vis_start_indices:
                    if start_idx > current_pos:
                        text_len = start_idx - current_pos
                        start_id = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                        llm_pos_ids_list.append(torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + start_id)
                    
                    vis_token = valid_ids[start_idx + 1]
                    
                    if vis_token == image_token_id:
                        t, h, w = image_grid_thw[img_idx]
                        img_idx += 1
                    elif vis_token == video_token_id:
                        t, h, w = video_grid_thw[vid_idx]
                        vid_idx += 1
                    else:
                        continue
                        
                    llm_t, llm_h, llm_w = t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size
                    vis_len = llm_t * llm_h * llm_w
                    
                    start_id = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    
                    t_idx = torch.arange(llm_t, device=device).view(-1, 1).expand(-1, llm_h * llm_w).flatten()
                    h_idx = torch.arange(llm_h, device=device).view(1, -1, 1).expand(llm_t, -1, llm_w).flatten()
                    w_idx = torch.arange(llm_w, device=device).view(1, 1, -1).expand(llm_t, llm_h, -1).flatten()
                    
                    llm_pos_ids_list.append(torch.stack([t_idx, h_idx, w_idx]) + start_id)
                    current_pos = start_idx + vis_len
                
                if current_pos < len(valid_ids):
                    text_len = len(valid_ids) - current_pos
                    start_id = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + start_id)

            flat_pos = torch.cat(llm_pos_ids_list, dim=1)
            if attention_mask is not None:
                position_ids[:, i, attention_mask[i] == 1] = flat_pos
            else:
                position_ids[:, i, :] = flat_pos

        return position_ids

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        sequence_lengths: list[int] | None = None,
        vision_slices_per_seq: list[list[dict]] | None = None,
        seq_image_indices: list[tuple[int, int]] | None = None,
        seq_vision_placeholders: list[list[tuple[int, int]]] | None = None,
    ) -> torch.Tensor:
        # Debug log for input_ids before processing
        if not hasattr(self, '_debug_input_ids_logged'):
            _debug_log(f"\n[DEBUG Qwen2VLForConditionalGeneration.forward] Input IDs:")
            if input_ids is not None:
                if isinstance(input_ids, torch.Tensor):
                    input_ids_flat = input_ids.view(-1) if input_ids.dim() > 1 else input_ids
                    _debug_log(f"  input_ids shape: {input_ids.shape}")
                    _debug_log(f"  input_ids dtype: {input_ids.dtype}")
                    _debug_log(f"  input_ids length: {len(input_ids_flat)}")
                    # Check values at key positions
                    if len(input_ids_flat) > 515:
                        _debug_log(f"  input_ids[514]: {input_ids_flat[514].item()}")
                        _debug_log(f"  input_ids[515]: {input_ids_flat[515].item()} (first seq last)")
                        if len(input_ids_flat) > 516:
                            _debug_log(f"  input_ids[516]: {input_ids_flat[516].item()} (second seq start)")
                    if len(input_ids_flat) > 941:
                        _debug_log(f"  input_ids[940]: {input_ids_flat[940].item()}")
                        _debug_log(f"  input_ids[941]: {input_ids_flat[941].item()} (second seq last)")
                    if len(input_ids_flat) > 3914:
                        _debug_log(f"  input_ids[3913]: {input_ids_flat[3913].item()}")
                        _debug_log(f"  input_ids[3914]: {input_ids_flat[3914].item()} (last seq last)")
                    # Check image_token_id
                    image_token_id = getattr(self.config, 'image_token_id', None)
                    if image_token_id is not None:
                        _debug_log(f"  image_token_id: {image_token_id}")
                        image_token_count = (input_ids_flat == image_token_id).sum().item()
                        _debug_log(f"  image_token_count in input_ids: {image_token_count}")
                        # Check if last token positions are image tokens
                        if len(input_ids_flat) > 515:
                            is_img_515 = input_ids_flat[515].item() == image_token_id
                            _debug_log(f"  input_ids[515] is image_token: {is_img_515}")
                        if len(input_ids_flat) > 941:
                            is_img_941 = input_ids_flat[941].item() == image_token_id
                            _debug_log(f"  input_ids[941] is image_token: {is_img_941}")
                else:
                    _debug_log(f"  input_ids type: {type(input_ids)}")
            else:
                _debug_log(f"  input_ids is None")
            if sequence_lengths:
                _debug_log(f"  sequence_lengths: {sequence_lengths}")
            self._debug_input_ids_logged = True
        
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
            if inputs_embeds.dim() == 3:
                inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
            
            # Debug log for embeddings after get_input_embeddings
            if not hasattr(self, '_debug_embeddings_after_get_logged'):
                _debug_log(f"\n[DEBUG Qwen2VLForConditionalGeneration.forward] After get_input_embeddings:")
                _debug_log(f"  inputs_embeds shape: {inputs_embeds.shape}")
                _debug_log(f"  inputs_embeds dtype: {inputs_embeds.dtype}")
                if inputs_embeds.numel() > 0:
                    _debug_log(f"  inputs_embeds min/max: {inputs_embeds.min().item():.4f} / {inputs_embeds.max().item():.4f}")
                    _debug_log(f"  inputs_embeds mean: {inputs_embeds.mean().item():.4f}")
                    _debug_log(f"  inputs_embeds all zeros: {(inputs_embeds == 0).all().item()}")
                    # Check last token positions
                    if sequence_lengths:
                        offsets = [0]
                        for length in sequence_lengths:
                            offsets.append(offsets[-1] + length)
                        for i, offset in enumerate(offsets[1:], 1):
                            last_idx = offset - 1
                            if last_idx >= 0 and last_idx < inputs_embeds.shape[0]:
                                _debug_log(f"  Sequence {i} last token at index {last_idx}: inputs_embeds[{last_idx}, :5] = {inputs_embeds[last_idx, :5].tolist()}")
                                if input_ids is not None:
                                    input_ids_flat = input_ids.view(-1) if input_ids.dim() > 1 else input_ids
                                    if last_idx < len(input_ids_flat):
                                        _debug_log(f"    input_ids[{last_idx}] = {input_ids_flat[last_idx].item()}")
                self._debug_embeddings_after_get_logged = True
        
        needs_clone = (pixel_values is not None and vision_slices_per_seq is None) or vision_slices_per_seq is not None
        if needs_clone: inputs_embeds = inputs_embeds.clone()
        
        vision_token_count = 0

        # --- Visual Replacement Logic ---
        if not hasattr(self, '_debug_vision_replacement_check_logged'):
            _debug_log(f"\n[DEBUG Qwen2VLForConditionalGeneration.forward] Vision replacement check:")
            _debug_log(f"  pixel_values is not None: {pixel_values is not None}")
            _debug_log(f"  seq_image_indices: {seq_image_indices}")
            _debug_log(f"  sequence_lengths: {sequence_lengths}")
            if seq_image_indices and sequence_lengths:
                _debug_log(f"  len(seq_image_indices) == len(sequence_lengths): {len(seq_image_indices) == len(sequence_lengths)}")
            self._debug_vision_replacement_check_logged = True
        
        if pixel_values is not None:
            # Check if we have seq_image_indices for multi-sequence mapping (like qwen3_vl)
            if seq_image_indices and sequence_lengths and len(seq_image_indices) == len(sequence_lengths):
                # Use seq_image_indices to correctly map images to sequences
                offsets = [0]
                for length in sequence_lengths:
                    offsets.append(offsets[-1] + length)
                
                image_embeds_list = self.visual(pixel_values, image_grid_thw)
                if isinstance(image_embeds_list, list):
                    image_embeddings = image_embeds_list
                else:
                    # Split tensor into list of per-image embeddings
                    num_images = image_embeds_list.shape[0]
                    image_embeddings = [image_embeds_list[i] for i in range(num_images)]
                
                # Process each sequence separately
                for seq_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                    seq_length = end - start
                    if seq_length <= 0:
                        continue
                    
                    # Get image index range for this sequence
                    img_start_idx, img_end_idx = seq_image_indices[seq_idx]
                    
                    # Collect all image embeddings for this sequence
                    seq_image_embeds = image_embeddings[img_start_idx:img_end_idx]
                    
                    if not seq_image_embeds:
                        continue
                    
                    # Concatenate all image embeddings for this sequence
                    seq_vision_tokens = torch.cat([
                        emb.to(inputs_embeds.device, inputs_embeds.dtype)
                        for emb in seq_image_embeds
                    ], dim=0)
                    
                    slice_len = seq_vision_tokens.size(0)
                    
                    # Use vision_placeholders to get correct target_offset (like normal path)
                    if seq_vision_placeholders and seq_idx < len(seq_vision_placeholders):
                        placeholders = seq_vision_placeholders[seq_idx]
                        if placeholders:
                            # Use the first placeholder's offset (typically there's only one)
                            target_offset, expected_length = placeholders[0]
                            if expected_length != slice_len:
                                # Warn but use actual length
                                pass
                            
                            target_start = start + target_offset
                            target_end = target_start + slice_len
                            
                            # Debug log before replacement
                            if not hasattr(self, '_debug_vision_replace_before_logged'):
                                _debug_log(f"\n[DEBUG Qwen2VLForConditionalGeneration.forward] Before vision replacement:")
                                _debug_log(f"  Sequence {seq_idx}: start={start}, end={end}, seq_length={seq_length}")
                                _debug_log(f"  target_offset={target_offset}, slice_len={slice_len}")
                                _debug_log(f"  target_start={target_start}, target_end={target_end}")
                                _debug_log(f"  last_token_idx={end-1}")
                                _debug_log(f"  Will replace inputs_embeds[{target_start}:{target_end}]")
                                if target_end > end - 1:
                                    _debug_log(f"  WARNING: target_end ({target_end}) > last_token_idx ({end-1})! This will overwrite last token!")
                                elif target_end == end - 1:
                                    _debug_log(f"  WARNING: target_end ({target_end}) == last_token_idx ({end-1})! This will overwrite last token!")
                                # Check last token before replacement
                                last_idx = end - 1
                                if last_idx >= 0 and last_idx < inputs_embeds.shape[0]:
                                    _debug_log(f"  inputs_embeds[{last_idx}, :5] before replacement: {inputs_embeds[last_idx, :5].tolist()}")
                                # Check target range before replacement
                                if target_start < inputs_embeds.shape[0] and target_end <= inputs_embeds.shape[0]:
                                    _debug_log(f"  inputs_embeds[{target_start}, :5] before replacement: {inputs_embeds[target_start, :5].tolist()}")
                                    if target_end > target_start + 1:
                                        _debug_log(f"  inputs_embeds[{target_end-1}, :5] before replacement: {inputs_embeds[target_end-1, :5].tolist()}")
                            if target_end > end:
                                raise ValueError(
                                    f"Visual token target range [{target_start}, {target_end}) "
                                    f"is out of sequence bounds [{start}, {end})"
                                )
                            
                            inputs_embeds[target_start:target_end] = seq_vision_tokens
                            vision_token_count += slice_len
                            
                            # Debug log after replacement
                            if not hasattr(self, '_debug_vision_replace_after_logged'):
                                _debug_log(f"\n[DEBUG Qwen2VLForConditionalGeneration.forward] After vision replacement:")
                                last_idx = end - 1
                                if last_idx >= 0 and last_idx < inputs_embeds.shape[0]:
                                    _debug_log(f"  inputs_embeds[{last_idx}, :5] after replacement: {inputs_embeds[last_idx, :5].tolist()}")
                                # Check target range after replacement
                                if target_start < inputs_embeds.shape[0] and target_end <= inputs_embeds.shape[0]:
                                    _debug_log(f"  inputs_embeds[{target_start}, :5] after replacement: {inputs_embeds[target_start, :5].tolist()}")
                                    if target_end > target_start + 1:
                                        _debug_log(f"  inputs_embeds[{target_end-1}, :5] after replacement: {inputs_embeds[target_end-1, :5].tolist()}")
                                self._debug_vision_replace_after_logged = True
                        else:
                            # No placeholders for this sequence - fallback to beginning
                            if start + slice_len > end:
                                raise ValueError(
                                    f"Visual tokens exceed sequence length: position {start} + "
                                    f"tokens {slice_len} > sequence end {end}"
                                )
                            inputs_embeds[start : start + slice_len] = seq_vision_tokens
                            vision_token_count += slice_len
                    else:
                        # No seq_vision_placeholders provided - fallback to beginning
                        if start + slice_len > end:
                            raise ValueError(
                                f"Visual tokens exceed sequence length: position {start} + "
                                f"tokens {slice_len} > sequence end {end}"
                            )
                        inputs_embeds[start : start + slice_len] = seq_vision_tokens
                        vision_token_count += slice_len
            else:
                # Fallback: original logic using masked_scatter (only works correctly for single sequence)
                # For multi-sequence scenarios, this logic is INCORRECT because it uses masked_scatter
                # on flattened input_ids without considering sequence boundaries.
                # In prefill-only mode, we should always have seq_image_indices and seq_vision_placeholders.
                if sequence_lengths and len(sequence_lengths) > 1:
                    raise ValueError(
                        f"Multi-sequence scenario detected ({len(sequence_lengths)} sequences) "
                        f"but seq_image_indices not provided. The fallback logic using masked_scatter "
                        f"on flattened input_ids is incorrect for multi-sequence scenarios. "
                        f"Please ensure seq_image_indices and seq_vision_placeholders are provided."
                    )
                
                image_embeds = self.visual(pixel_values, image_grid_thw)
                if isinstance(image_embeds, list):
                    image_embeds = torch.cat(image_embeds, dim=0).to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                else:
                    image_embeds = image_embeds.view(-1, image_embeds.shape[-1]).to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )

                if input_ids.dim() == 2:
                    input_ids_flat = input_ids.view(-1)
                else:
                    input_ids_flat = input_ids

                image_token_id = self.config.image_token_id
                is_image_token = input_ids_flat == image_token_id
                image_mask = is_image_token.unsqueeze(-1).expand_as(inputs_embeds)

                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                vision_token_count = is_image_token.sum().item()
        
        # Debug log after vision token replacement
        if not hasattr(self, '_debug_vision_replacement_logged') and vision_token_count > 0:
            _debug_log(f"\n[DEBUG Qwen2VLForConditionalGeneration.forward] After vision replacement:")
            _debug_log(f"  vision_token_count: {vision_token_count}")
            _debug_log(f"  inputs_embeds shape: {inputs_embeds.shape}")
            if sequence_lengths:
                _debug_log(f"  sequence_lengths: {sequence_lengths}")
                # Calculate actual last token positions after vision replacement
                offsets = [0]
                for length in sequence_lengths:
                    offsets.append(offsets[-1] + length)
                _debug_log(f"  offsets after vision replacement: {offsets}")
                # Check last token positions
                for i, offset in enumerate(offsets[1:], 1):
                    last_idx = offset - 1
                    if last_idx >= 0 and last_idx < inputs_embeds.shape[0]:
                        _debug_log(f"  Sequence {i} last token at index {last_idx}: inputs_embeds[{last_idx}, :5] = {inputs_embeds[last_idx, :5].tolist()}")
            self._debug_vision_replacement_logged = True

        # Debug log for inputs_embeds before forward pass
        if not hasattr(self, '_debug_inputs_embeds_logged'):
            _debug_log(f"\n[DEBUG Qwen2VLForConditionalGeneration.forward] Inputs embeds:")
            _debug_log(f"  inputs_embeds shape: {inputs_embeds.shape}")
            _debug_log(f"  inputs_embeds dtype: {inputs_embeds.dtype}")
            if inputs_embeds.numel() > 0:
                _debug_log(f"  inputs_embeds min/max: {inputs_embeds.min().item():.4f} / {inputs_embeds.max().item():.4f}")
                _debug_log(f"  inputs_embeds mean: {inputs_embeds.mean().item():.4f}")
                _debug_log(f"  inputs_embeds has NaN: {torch.isnan(inputs_embeds).any().item()}")
                _debug_log(f"  inputs_embeds has Inf: {torch.isinf(inputs_embeds).any().item()}")
                _debug_log(f"  inputs_embeds all zeros: {(inputs_embeds == 0).all().item()}")
                # Check values at key positions (last tokens of sequences)
                if inputs_embeds.shape[0] > 515:
                    _debug_log(f"  inputs_embeds[515, :5]: {inputs_embeds[515, :5].tolist()}")
                if inputs_embeds.shape[0] > 941:
                    _debug_log(f"  inputs_embeds[941, :5]: {inputs_embeds[941, :5].tolist()}")
                if inputs_embeds.shape[0] > 1208:
                    _debug_log(f"  inputs_embeds[1208, :5]: {inputs_embeds[1208, :5].tolist()}")
                if inputs_embeds.shape[0] > 3914:
                    _debug_log(f"  inputs_embeds[3914, :5]: {inputs_embeds[3914, :5].tolist()}")
                # Check a few positions before and after last tokens
                if inputs_embeds.shape[0] > 516:
                    _debug_log(f"  inputs_embeds[514, :5]: {inputs_embeds[514, :5].tolist()} (before last)")
                    _debug_log(f"  inputs_embeds[516, :5]: {inputs_embeds[516, :5].tolist()} (after last)")
            _debug_log(f"  vision_token_count: {vision_token_count}")
            if sequence_lengths:
                _debug_log(f"  sequence_lengths: {sequence_lengths}")
            self._debug_inputs_embeds_logged = True

        # --- M-RoPE Position Calculation ---
        # Logic: 
        # 1. If grids are present, recalculate positions using 3D logic.
        # 2. If no grids (text-only), fallback to expanding existing 1D positions to 3D.
        
        if image_grid_thw is not None:
            # Reshape input_ids to batch if needed for get_rope_index
            if input_ids.dim() == 1 and sequence_lengths:
                input_ids_batch = input_ids.unsqueeze(0)
            else:
                input_ids_batch = input_ids

            # positions_3d shape: [3, batch, seq]
            positions = self.get_rope_index(input_ids_batch, image_grid_thw, None)
            
        else:
            # Fallback: Force 1D/2D positions to 3D M-RoPE format [3, batch, seq]
            if positions is not None:
                if positions.dim() == 1:
                    # Case: Flattened [seq] -> Expand to [3, 1, seq]
                    # This assumes batch_size=1, which is typical for flattened inference in this framework
                    positions = positions.unsqueeze(0).unsqueeze(0).expand(3, -1, -1)
                elif positions.dim() == 2:
                    # Case: Batched [batch, seq] -> Expand to [3, batch, seq]
                    positions = positions.unsqueeze(0).expand(3, -1, -1)
                # If already 3D, do nothing

        # --- Forward Pass ---
        hidden_states = self.language_model(
            input_ids=None,
            positions=positions, # Must be 3D [3, batch, seq]
            inputs_embeds=inputs_embeds,
        )

        # Debug log for hidden_states
        if not hasattr(self, '_debug_hidden_states_logged'):
            _debug_log(f"\n[DEBUG Qwen2VLForConditionalGeneration.forward] Hidden states:")
            _debug_log(f"  hidden_states shape: {hidden_states.shape}")
            _debug_log(f"  hidden_states dtype: {hidden_states.dtype}")
            if hidden_states.numel() > 0:
                _debug_log(f"  hidden_states min/max: {hidden_states.min().item():.4f} / {hidden_states.max().item():.4f}")
                _debug_log(f"  hidden_states mean: {hidden_states.mean().item():.4f}")
                _debug_log(f"  hidden_states has NaN: {torch.isnan(hidden_states).any().item()}")
                _debug_log(f"  hidden_states has Inf: {torch.isinf(hidden_states).any().item()}")
                _debug_log(f"  hidden_states all zeros: {(hidden_states == 0).all().item()}")
                # Check values at key positions (last tokens of sequences)
                if hidden_states.shape[0] > 515:
                    _debug_log(f"  hidden_states[515, :5]: {hidden_states[515, :5].tolist()}")
                if hidden_states.shape[0] > 941:
                    _debug_log(f"  hidden_states[941, :5]: {hidden_states[941, :5].tolist()}")
                if hidden_states.shape[0] > 1208:
                    _debug_log(f"  hidden_states[1208, :5]: {hidden_states[1208, :5].tolist()}")
                if hidden_states.shape[0] > 3914:
                    _debug_log(f"  hidden_states[3914, :5]: {hidden_states[3914, :5].tolist()}")
                # Check a few positions before and after last tokens
                if hidden_states.shape[0] > 516:
                    _debug_log(f"  hidden_states[514, :5]: {hidden_states[514, :5].tolist()} (before last)")
                    _debug_log(f"  hidden_states[516, :5]: {hidden_states[516, :5].tolist()} (after last)")
            if sequence_lengths:
                _debug_log(f"  sequence_lengths: {sequence_lengths}")
            self._debug_hidden_states_logged = True

        return hidden_states

    def compute_logits(self, hidden_states):
        return self.language_model.compute_logits(hidden_states)


def load_qwen2_vl_model(model_path, config):
    # Same loading logic as before, ensuring name_mapping handles gate_up_proj
    hf_config = config.hf_config
    model = Qwen2VLForConditionalGeneration(hf_config)
    
    from nanovllm.utils.loader import load_model
    
    # ... (Copy the name_mapping function from your previous code) ...
    # ... (Copy the loading logic) ...
    
    # Re-insert the provided loading logic here
    # Ensure name_mapping maps "model.layers..." correctly
    def name_mapping(weight_name: str) -> str | None:
        if weight_name.startswith("model.language_model."):
            sub = weight_name[len("model.language_model."):]
            if sub.startswith("model."): sub = sub[len("model."):]
            return "language_model.model." + sub
        if weight_name.startswith("model.visual."):
            sub = weight_name[len("model.visual."):]
            return "visual.vision." + sub
        if weight_name.startswith("visual."):
            sub = weight_name[len("visual."):]
            return "visual.vision." + sub if not sub.startswith("vision.") else weight_name
        if weight_name.startswith("lm_head."):
            return "language_model.lm_head." + weight_name[len("lm_head."):]
        return None

    load_model(model, model_path, name_mapping=name_mapping)
    return model


def create_vision_model(config, **kwargs):
    return Qwen2VisionEncoder(config)


__all__ = ["Qwen2VLForConditionalGeneration", "load_qwen2_vl_model"]
