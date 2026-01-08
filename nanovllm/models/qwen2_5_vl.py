"""Simplified implementation of the Qwen2_5-VL multimodal model.

This module inlines both the text backbone and the vision encoder to minimise
changes to other components.
Reference: transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope


# ---------------------------------------------------------------------------
# Text backbone
# ---------------------------------------------------------------------------


class Qwen2_5VLTextAttention(nn.Module):

    def __init__(
        self,
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
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

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
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Qwen2_5 does not use q_norm/k_norm

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # Qwen2_5 does not use q_norm/k_norm
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen2_5VLTextMLP(nn.Module):

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


class Qwen2_5VLTextDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_scaling = None

        self.self_attn = Qwen2_5VLTextAttention(
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
        self.mlp = Qwen2_5VLTextMLP(
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


class Qwen2_5VLTextModel(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen2_5VLTextDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        vision_token_count: int | None = None,
        visual_pos_mask: torch.Tensor | None = None,
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


class Qwen2_5VLTextForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen2_5VLTextModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        vision_token_count: int | None = None,
        visual_pos_mask: torch.Tensor | None = None,
        **_: dict,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            vision_token_count=vision_token_count,
            visual_pos_mask=visual_pos_mask,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------------------------


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1 + torch.tanh(inner))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Qwen2_5-VL vision RoPE application (similar to transformers implementation)
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = getattr(config, "temporal_patch_size", 2)
        self.in_channels = getattr(config, "in_channels", 3)
        # Qwen2.5-VL vision uses hidden_size (not embed_dim)
        embed_dim = config.hidden_size
        self.embed_dim = embed_dim

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Qwen2.5-VL patch embed forward (similar to transformers)
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2_5_VLMLP(nn.Module):
    """Qwen2.5-VL uses gate_up_proj structure (similar to Mistral)"""
    def __init__(self, config, bias: bool = True) -> None:
        super().__init__()
        # Qwen2.5-VL vision uses hidden_size
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act == "silu":
            self.act_fn = F.silu
        elif hidden_act == "gelu":
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")
        else:
            # Default to SiLU if unknown activation
            self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # gate_up_proj style: SiLU(gate) * up
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(self.act_fn(gate) * up)
        return hidden_states


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # Qwen2.5-VL vision uses hidden_size (not embed_dim)
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        # Qwen2_5-VL vision attention implementation (similar to transformers)
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Simple eager attention for now (can be optimized with flash attention later)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max() if cu_seqlens.numel() > 1 else seq_length
        
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, config, out_hidden_size=None) -> None:
        super().__init__()
        # Following transformers official implementation:
        # - context_dim = config.hidden_size (vision encoder internal dimension)
        # - output_dim = config.out_hidden_size (should match text_config.hidden_size)
        context_dim = config.hidden_size
        self.hidden_size = context_dim * (config.spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)  # Use RMSNorm like Qwen2.5
        
        # Output dimension: use config.out_hidden_size (as per transformers)
        # If out_hidden_size is provided explicitly, use it (for compatibility)
        if out_hidden_size is not None:
            output_dim = out_hidden_size
        else:
            # Use config.out_hidden_size if available, otherwise fallback to config.hidden_size
            output_dim = getattr(config, "out_hidden_size", config.hidden_size)
        
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_dim, bias=True),
        )
        self.merge_size = config.spatial_merge_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.isnan(hidden_states).any():
            logger.error(f"[Qwen2_5_VLPatchMerger] ERROR: Input contains NaN!")
        
        hidden_states = self.ln_q(hidden_states)
        if torch.isnan(hidden_states).any():
            logger.error(f"[Qwen2_5_VLPatchMerger] ERROR: After ln_q contains NaN!")
        
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.mlp(hidden_states)
        
        if torch.isnan(hidden_states).any():
            logger.error(f"[Qwen2_5_VLPatchMerger] ERROR: Output contains NaN!")
            logger.error(f"[Qwen2_5_VLPatchMerger] MLP layers: {[type(l).__name__ for l in self.mlp]}")
        
        return hidden_states


class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # Qwen2.5-VL uses hidden_size and RMSNorm
        self.norm1 = RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=1e-6)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.attn(hidden_states, cu_seqlens, position_embeddings, **kwargs)
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen2_5_VLVisionModel(nn.Module):
    def __init__(self, config, out_hidden_size=None) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Qwen2_5_VisionPatchEmbed(config)
        # Qwen2.5-VL vision uses hidden_size (not embed_dim)
        self.hidden_size = config.hidden_size
        # Qwen2.5-VL does not use pos_embed, it uses rot_pos_emb directly
        head_dim = self.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5_VLPatchMerger(config=config, out_hidden_size=out_hidden_size)
        # Qwen2.5-VL does not support DeepStack

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h = height // merge_size
            merged_w = width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row.view(1, 1, -1, 1)
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col.view(1, 1, 1, -1)
            )

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset: offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # Qwen2.5-VL vision forward
        pixel_values = pixel_values.to(self.patch_embed.proj.weight.dtype)
        seq_tokens = self.patch_embed(pixel_values)
        hidden_states = seq_tokens.reshape(-1, self.hidden_size)

        # Qwen2_5-VL uses rot_pos_emb directly for position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(hidden_states.size(0), -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings=position_embeddings)
            if torch.isnan(hidden_states).any():
                logger.error(f"[Qwen2_5_VLVisionModel] ERROR: NaN after block {i}!")
                break

        if torch.isnan(hidden_states).any():
            logger.error(f"[Qwen2_5_VLVisionModel] ERROR: NaN before merger!")

        hidden_states = self.merger(hidden_states)
        
        if torch.isnan(hidden_states).any():
            logger.error(f"[Qwen2_5_VLVisionModel] ERROR: NaN after merger!")
        
        return hidden_states


class Qwen2_5_VisionEncoder(nn.Module):
    def __init__(self, vision_config, out_hidden_size=None) -> None:
        super().__init__()
        self.config = vision_config
        self.vision = Qwen2_5_VLVisionModel(vision_config, out_hidden_size=out_hidden_size)

    def _linear_patch_embed(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        proj = self.vision.patch_embed.proj
        weight = proj.weight.view(proj.out_channels, -1)
        bias = proj.bias
        # Ensure patch_tokens has the same dtype as weight
        patch_tokens = patch_tokens.to(weight.dtype)
        return torch.nn.functional.linear(patch_tokens, weight, bias)

    def _run_vision_from_tokens(
        self,
        token_list: list[torch.Tensor],
        grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        # Qwen2_5-VL does not support DeepStack
        # Process tokens through vision model
        proj = self.vision.patch_embed.proj
        device = proj.weight.device
        dtype = proj.weight.dtype
        
        tokens = torch.cat([t.to(device=device, dtype=dtype) for t in token_list], dim=0)
        grids = grid_thw.to(device=device, dtype=torch.int32)
        
        # Process through vision blocks
        hidden_states = tokens
        
        rotary_pos_emb = self.vision.rot_pos_emb(grids).to(hidden_states.dtype)
        rotary_pos_emb = rotary_pos_emb.reshape(hidden_states.size(0), -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Calculate cu_seqlens for Flash Attention Varlen
        # For each image, we have grids[:, 0] frames, each with grids[:, 1] * grids[:, 2] spatial tokens
        spatial_tokens_per_frame = grids[:, 1] * grids[:, 2]  # [h * w for each image]
        cu_seqlens = torch.repeat_interleave(spatial_tokens_per_frame, grids[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        # Debug: log cu_seqlens calculation
        if not hasattr(self, '_debug_cu_seqlens_logged'):
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"=== CU Sequence Lengths Calculation ==="
            )
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"grids shape: {grids.shape}, values:\n{grids.cpu().tolist()}"
            )
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"spatial_tokens_per_frame: {spatial_tokens_per_frame.cpu().tolist()}"
            )
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"grids[:, 0] (temporal): {grids[:, 0].cpu().tolist()}"
            )
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"cu_seqlens (before pad): {cu_seqlens[1:].cpu().tolist()}"
            )
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"cu_seqlens (after pad): {cu_seqlens.cpu().tolist()}"
            )
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"hidden_states shape before blocks: {hidden_states.shape}"
            )
            self._debug_cu_seqlens_logged = True

        for block in self.vision.blocks:
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings=position_embeddings)

        hidden_states = self.vision.merger(hidden_states)
        
        # Debug: log after merger
        if not hasattr(self, '_debug_after_merger_logged'):
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"hidden_states shape after merger: {hidden_states.shape}"
            )
            spatial_merge_size = self.config.spatial_merge_size
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"spatial_merge_size: {spatial_merge_size}"
            )
            self._debug_after_merger_logged = True

        split_sizes = (grids.prod(-1) // (self.config.spatial_merge_size**2)).tolist()
        image_chunks = list(torch.split(hidden_states, split_sizes))
        
        # Debug: log split results
        if not hasattr(self, '_debug_split_sizes_logged'):
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"split_sizes: {split_sizes}"
            )
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"image_chunks lengths: {[chunk.shape[0] for chunk in image_chunks]}"
            )
            print(
                f"[DEBUG qwen2.5vl._run_vision_from_tokens] "
                f"Total hidden_states: {hidden_states.shape[0]}, sum of split_sizes: {sum(split_sizes)}"
            )
            self._debug_split_sizes_logged = True

        return image_chunks

    def _normalize_pixel_inputs(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        in_channels = getattr(self.config, "in_channels", 3)
        num_dims = pixel_values.dim()

        channel_axis = None
        for axis in range(1, num_dims - 2):
            if pixel_values.shape[axis] == in_channels:
                channel_axis = axis
                break
        if channel_axis is None:
            channel_axis = 1

        permute_order = [0, channel_axis]
        temporal_axes = [
            axis for axis in range(1, num_dims - 2) if axis != channel_axis
        ]
        permute_order.extend(temporal_axes)
        permute_order.extend([num_dims - 2, num_dims - 1])

        pixel_values = pixel_values.permute(*permute_order).contiguous()

        batch = pixel_values.shape[0]
        channels = pixel_values.shape[1]
        height = pixel_values.shape[-2]
        width = pixel_values.shape[-1]

        temporal = int(math.prod(pixel_values.shape[2:-2]))
        pixel_values = pixel_values.reshape(
            batch,
            channels,
            temporal,
            height,
            width,
        )

        return pixel_values, batch, temporal, height, width

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pixel_values.dim() <= 3:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required for flattened inputs")
            grids = image_grid_thw.to(pixel_values.device).to(torch.int64)
            tokens_per_image = grids.prod(-1).tolist()
            if pixel_values.dim() == 3:
                batch, tokens, feature = pixel_values.shape
                flat = pixel_values.reshape(batch * tokens, feature)
            else:
                flat = pixel_values

            # Debug: log splitting details
            if not hasattr(self, '_debug_vision_encoder_split_logged'):
                print(
                    f"[DEBUG qwen2.5vl.vision.forward (flattened)] "
                    f"=== Vision Encoder Flattened Input Processing ==="
                )
                print(
                    f"[DEBUG qwen2.5vl.vision.forward (flattened)] "
                    f"pixel_values shape: {pixel_values.shape}, dim: {pixel_values.dim()}"
                )
                print(
                    f"[DEBUG qwen2.5vl.vision.forward (flattened)] "
                    f"image_grid_thw shape: {image_grid_thw.shape}, values:\n{image_grid_thw.cpu().tolist()}"
                )
                print(
                    f"[DEBUG qwen2.5vl.vision.forward (flattened)] "
                    f"tokens_per_image: {tokens_per_image} (total: {sum(tokens_per_image)})"
                )
                print(
                    f"[DEBUG qwen2.5vl.vision.forward (flattened)] "
                    f"flat shape: {flat.shape} (expected: [{sum(tokens_per_image)}, {feature if pixel_values.dim() == 3 else pixel_values.shape[1]}])"
                )
                self._debug_vision_encoder_split_logged = True

            # Ensure pixel_values have the correct dtype
            proj_dtype = self.vision.patch_embed.proj.weight.dtype
            flat = flat.to(proj_dtype)
            
            splits = torch.split(flat, tokens_per_image, dim=0)
            token_list = [self._linear_patch_embed(chunk) for chunk in splits]
            
            # Debug: log split results
            if not hasattr(self, '_debug_vision_encoder_splits_logged'):
                print(
                    f"[DEBUG qwen2.5vl.vision.forward (flattened)] "
                    f"Split into {len(splits)} chunks, token_list lengths: {[t.shape[0] for t in token_list]}"
                )
                self._debug_vision_encoder_splits_logged = True

            return self._run_vision_from_tokens(token_list, grids)

        pixel_values, batch, temporal, height, width = self._normalize_pixel_inputs(
            pixel_values
        )
        # Ensure pixel_values have the correct dtype
        pixel_values = pixel_values.to(self.vision.patch_embed.proj.weight.dtype)

        if image_grid_thw is None:
            grid = torch.tensor(
                [
                    [
                        temporal,
                        height // self.config.patch_size,
                        width // self.config.patch_size,
                    ]
                ]
                * batch,
                device=pixel_values.device,
                dtype=torch.int32,
            )
            image_grid_thw = grid
        else:
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            image_grid_thw = image_grid_thw.to(device=pixel_values.device, dtype=torch.int32)

        image_embeds = self.vision(pixel_values, image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // (self.config.spatial_merge_size**2)
        ).tolist()

        image_chunks = torch.split(image_embeds, split_sizes)
        image_tokens = torch.stack(list(image_chunks), dim=0)

        return image_tokens


def create_vision_model(config, out_hidden_size=None, **kwargs) -> nn.Module:
    del kwargs
    return Qwen2_5_VisionEncoder(config, out_hidden_size=out_hidden_size)


def get_vision_model(config, **kwargs) -> nn.Module:
    return create_vision_model(config, **kwargs)


# ---------------------------------------------------------------------------
# Multimodal wrapper
# ---------------------------------------------------------------------------


class Qwen2_5VLForConditionalGeneration(nn.Module):
    """Qwen2_5-VL conditional generation model (language + vision)."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_config = getattr(config, "text_config", config)
        self.vision_config = getattr(config, "vision_config", None)

        if self.vision_config is None:
            raise ValueError("vision_config is missing; cannot build a multimodal model")

        # Following transformers official implementation:
        # merger outputs config.out_hidden_size, which should equal text_config.hidden_size
        vision_hidden_size = getattr(self.vision_config, "hidden_size", None)
        vision_out_hidden_size = getattr(self.vision_config, "out_hidden_size", None)
        text_hidden_size = getattr(self.text_config, "hidden_size", None)
        
        logger.debug(f"[Qwen2_5VLForConditionalGeneration] Vision config hidden_size: {vision_hidden_size}")
        logger.debug(f"[Qwen2_5VLForConditionalGeneration] Vision config out_hidden_size: {vision_out_hidden_size}")
        logger.debug(f"[Qwen2_5VLForConditionalGeneration] Text config hidden_size: {text_hidden_size}")
        
        # Use vision_config.out_hidden_size for merger output (as per transformers)
        # If out_hidden_size is not set, use text_config.hidden_size as fallback
        if vision_out_hidden_size is None:
            if text_hidden_size is not None:
                logger.debug(f"[Qwen2_5VLForConditionalGeneration] Warning: vision_config.out_hidden_size not found, "
                      f"using text_config.hidden_size={text_hidden_size} as fallback")
                vision_out_hidden_size = text_hidden_size
            else:
                logger.debug(f"[Qwen2_5VLForConditionalGeneration] Warning: Neither vision_config.out_hidden_size "
                      f"nor text_config.hidden_size found, using vision_config.hidden_size={vision_hidden_size}")
                vision_out_hidden_size = vision_hidden_size
        
        self.visual = create_vision_model(
            self.vision_config, out_hidden_size=vision_out_hidden_size
        )
        self.language_model = Qwen2_5VLTextForCausalLM(self.text_config)

        logger.debug("[Qwen2_5VLForConditionalGeneration] Initialization complete")
        logger.debug(f"  - Vision encoder: {type(self.visual).__name__}")
        logger.debug(f"  - Language model: {type(self.language_model).__name__}")
    
        # packed_modules_mapping only applies to language_model, not vision
        # Vision model uses separate gate_proj/up_proj, not merged gate_up_proj
        # Loader checks "k in weight_name" on the ORIGINAL weight_name (before name_mapping)
        # Use simple keys that match the pattern in weight names (e.g., ".mlp.gate_proj" matches "model.layers.0.mlp.gate_proj")
        # Language model uses "self_attn" while vision uses "attn", so "self_attn" won't match vision
        # The loader will handle the replacement with layer numbers
        self.packed_modules_mapping = {
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".self_attn.q_proj": (".self_attn.qkv_proj", "q"),
            ".self_attn.k_proj": (".self_attn.qkv_proj", "k"),
            ".self_attn.v_proj": (".self_attn.qkv_proj", "v"),
        }
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.model.embed_tokens(input_ids)
    
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
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids and inputs_embeds cannot be None simultaneously")
            inputs_embeds = self.get_input_embeddings(input_ids)
            # Flatten if input_ids is in batch format [batch_size, seq_len]
            if inputs_embeds.dim() == 3:
                batch_size, seq_len, hidden_size = inputs_embeds.shape
                inputs_embeds = inputs_embeds.view(-1, hidden_size)

        total_tokens = inputs_embeds.size(0)
        inputs_embeds = inputs_embeds.clone()

        visual_pos_mask = torch.zeros(
            total_tokens, dtype=torch.bool, device=inputs_embeds.device
        )
        vision_token_count = 0

        if vision_slices_per_seq:
            if sequence_lengths is None:
                raise ValueError("sequence_lengths must be provided to align visual features")
            if len(sequence_lengths) != len(vision_slices_per_seq):
                raise ValueError("sequence_lengths and vision_slices_per_seq have different lengths")
            if sum(sequence_lengths) != total_tokens:
                raise ValueError("sum of sequence_lengths does not match total input tokens")

            offsets = [0]
            for length in sequence_lengths:
                offsets.append(offsets[-1] + length)

            for seq_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                seq_slices = vision_slices_per_seq[seq_idx]
                if not seq_slices:
                    continue

                for slice_info in seq_slices:
                    token_slice = slice_info["tokens"].to(
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
                    length = slice_info["length"]
                    target_offset = slice_info["target_offset"]

                    if token_slice.size(0) != length:
                        raise ValueError("Visual token slice length does not match the declared length")

                    target_start = start + target_offset
                    target_end = target_start + length
                    if target_end > end:
                        raise ValueError("Visual token target range is out of bounds")

                    inputs_embeds[target_start:target_end] = token_slice
                    visual_pos_mask[target_start:target_end] = True
                    vision_token_count += length
        elif pixel_values is not None:
            # Fallback path: process raw images when slices are not provided (legacy compatibility)
            if input_ids is None:
                raise ValueError("input_ids are required to locate visual placeholders")
            if sequence_lengths is None:
                raise ValueError("sequence_lengths are required to align visual features")
            if sum(sequence_lengths) != total_tokens:
                raise ValueError("sum of sequence_lengths does not match total input tokens")

            offsets = [0]
            for length in sequence_lengths:
                offsets.append(offsets[-1] + length)

            # Debug: log inputs before vision encoder
            if not hasattr(self, '_debug_qwen2_5vl_fallback_start_logged'):
                print(
                    f"[DEBUG qwen2.5vl.forward (fallback)] "
                    f"=== Fallback Path Inputs ==="
                )
                print(
                    f"[DEBUG qwen2.5vl.forward (fallback)] "
                    f"pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}, "
                    f"device: {pixel_values.device}"
                )
                print(
                    f"[DEBUG qwen2.5vl.forward (fallback)] "
                    f"image_grid_thw shape: {image_grid_thw.shape if image_grid_thw is not None else None}, "
                    f"dtype: {image_grid_thw.dtype if image_grid_thw is not None else None}"
                )
                if image_grid_thw is not None:
                    if image_grid_thw.dim() == 2:
                        print(
                            f"[DEBUG qwen2.5vl.forward (fallback)] "
                            f"image_grid_thw values:\n{image_grid_thw.cpu().tolist()}"
                        )
                    else:
                        print(
                            f"[DEBUG qwen2.5vl.forward (fallback)] "
                            f"image_grid_thw values: {image_grid_thw.cpu().tolist()}"
                        )
                print(
                    f"[DEBUG qwen2.5vl.forward (fallback)] "
                    f"sequence_lengths: {sequence_lengths}, total_tokens: {total_tokens}, offsets: {offsets}"
                )
                if seq_image_indices:
                    print(
                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                        f"seq_image_indices: {seq_image_indices}"
                    )
                if seq_vision_placeholders:
                    print(
                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                        f"seq_vision_placeholders: {seq_vision_placeholders}"
                    )
                print(
                    f"[DEBUG qwen2.5vl.forward (fallback)] "
                    f"inputs_embeds shape: {inputs_embeds.shape}, dtype: {inputs_embeds.dtype}, "
                    f"device: {inputs_embeds.device}"
                )
                # Log first few values of inputs_embeds before vision replacement
                if inputs_embeds.numel() > 0:
                    print(
                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                        f"inputs_embeds[0:5, 0:3] (first 5 tokens, first 3 dims):\n"
                        f"{inputs_embeds[0:5, 0:3].cpu().tolist()}"
                    )
                self._debug_qwen2_5vl_fallback_start_logged = True

            # Get image embeddings from vision encoder
            image_tokens = self.visual(pixel_values, image_grid_thw)
            # image_tokens can be either:
            # - tensor [batch_size, num_tokens, hidden_size] (when all images have same token count)
            # - list of tensors [num_tokens_i, hidden_size] (when images have different token counts)
            
            # Convert to list format for easier processing
            if isinstance(image_tokens, list):
                image_embeddings = image_tokens
                if not hasattr(self, '_debug_qwen2_5vl_image_tokens_logged'):
                    print(
                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                        f"=== Vision Encoder Output ==="
                    )
                    print(
                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                        f"vision encoder returned list of {len(image_embeddings)} tensors"
                    )
                    for i, emb in enumerate(image_embeddings):
                        print(
                            f"[DEBUG qwen2.5vl.forward (fallback)] "
                            f"Image {i}: shape={emb.shape}, dtype={emb.dtype}, "
                            f"device={emb.device}, "
                            f"mean={emb.mean().item():.6f}, std={emb.std().item():.6f}, "
                            f"min={emb.min().item():.6f}, max={emb.max().item():.6f}"
                        )
                        if i < 3:  # Print first few values for first 3 images
                            print(
                                f"[DEBUG qwen2.5vl.forward (fallback)] "
                                f"Image {i} embedding[0, 0:5] (first token, first 5 dims):\n"
                                f"{emb[0, 0:5].cpu().tolist()}"
                            )
                    self._debug_qwen2_5vl_image_tokens_logged = True
            else:
                # Split tensor into list of per-image embeddings
                num_images = image_tokens.shape[0]
                image_embeddings = [image_tokens[i] for i in range(num_images)]
                if not hasattr(self, '_debug_qwen2_5vl_image_tokens_logged'):
                    print(
                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                        f"=== Vision Encoder Output ==="
                    )
                    print(
                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                        f"vision encoder returned tensor shape {image_tokens.shape}, "
                        f"split into {len(image_embeddings)} embeddings"
                    )
                    for i, emb in enumerate(image_embeddings):
                        print(
                            f"[DEBUG qwen2.5vl.forward (fallback)] "
                            f"Image {i}: shape={emb.shape}, dtype={emb.dtype}, "
                            f"device={emb.device}, "
                            f"mean={emb.mean().item():.6f}, std={emb.std().item():.6f}"
                        )
                        if i < 3:  # Print first few values for first 3 images
                            print(
                                f"[DEBUG qwen2.5vl.forward (fallback)] "
                                f"Image {i} embedding[0, 0:5] (first token, first 5 dims):\n"
                                f"{emb[0, 0:5].cpu().tolist()}"
                            )
                    self._debug_qwen2_5vl_image_tokens_logged = True
            
            total_replaced = 0
            
            # For multi-sequence scenarios, we MUST use seq_image_indices and seq_vision_placeholders
            # The original fallback logic (scanning flattened input_ids) is incorrect for multi-sequence
            # because it doesn't account for sequence boundaries in the flattened input_ids.
            if seq_image_indices and len(seq_image_indices) == len(sequence_lengths):
                if not hasattr(self, '_debug_qwen2_5vl_mapping_logged'):
                    print(
                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                        f"Using seq_image_indices mapping for {len(sequence_lengths)} sequences"
                    )
                    self._debug_qwen2_5vl_mapping_logged = True
                # Use seq_image_indices to correctly map images to sequences
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
                    if slice_len > seq_length:
                        raise ValueError(
                            f"Visual tokens ({slice_len}) exceed sequence length ({seq_length}) "
                            f"for sequence {seq_idx}"
                        )
                    
                    # Use vision_placeholders to get correct target_offset (like normal path)
                    if seq_vision_placeholders and seq_idx < len(seq_vision_placeholders):
                        placeholders = seq_vision_placeholders[seq_idx]
                        if placeholders:
                            # Use the first placeholder's offset (typically there's only one)
                            if len(placeholders) == 1:
                                target_offset, expected_length = placeholders[0]
                                if expected_length != slice_len:
                                    # Warn about mismatch
                                    if not hasattr(self, f'_debug_length_mismatch_{seq_idx}_logged'):
                                        print(
                                            f"[DEBUG qwen2.5vl.forward (fallback)] "
                                            f"WARNING: Sequence {seq_idx} expected_length ({expected_length}) "
                                            f"!= actual slice_len ({slice_len}), using actual length"
                                        )
                                        setattr(self, f'_debug_length_mismatch_{seq_idx}_logged', True)
                                
                                target_start = start + target_offset
                                target_end = target_start + slice_len
                                if target_end > end:
                                    raise ValueError(
                                        f"Visual token target range [{target_start}, {target_end}) "
                                        f"is out of sequence bounds [{start}, {end})"
                                    )
                                
                                if not hasattr(self, f'_debug_placement_{seq_idx}_logged'):
                                    print(
                                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                                        f"=== Sequence {seq_idx} Vision Token Placement ==="
                                    )
                                    print(
                                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                                        f"Sequence {seq_idx}: placing {slice_len} vision tokens "
                                        f"at offset {target_offset} (absolute position {target_start}:{target_end}), "
                                        f"images [{img_start_idx}:{img_end_idx}]"
                                    )
                                    print(
                                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                                        f"Sequence {seq_idx}: seq_vision_tokens shape={seq_vision_tokens.shape}, "
                                        f"dtype={seq_vision_tokens.dtype}, device={seq_vision_tokens.device}"
                                    )
                                    print(
                                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                                        f"Sequence {seq_idx}: seq_vision_tokens[0, 0:5] (first token, first 5 dims):\n"
                                        f"{seq_vision_tokens[0, 0:5].cpu().tolist()}"
                                    )
                                    # Log inputs_embeds before replacement
                                    print(
                                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                                        f"Sequence {seq_idx}: inputs_embeds[{target_start}:{target_start+3}, 0:3] "
                                        f"BEFORE replacement:\n"
                                        f"{inputs_embeds[target_start:target_start+3, 0:3].cpu().tolist()}"
                                    )
                                    setattr(self, f'_debug_placement_{seq_idx}_logged', True)
                                
                                inputs_embeds[target_start:target_end] = seq_vision_tokens
                                visual_pos_mask[target_start:target_end] = True
                                total_replaced += slice_len
                                
                                # Log inputs_embeds after replacement (only for first sequence)
                                if seq_idx == 0 and not hasattr(self, f'_debug_after_replacement_{seq_idx}_logged'):
                                    print(
                                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                                        f"Sequence {seq_idx}: inputs_embeds[{target_start}:{target_start+3}, 0:3] "
                                        f"AFTER replacement:\n"
                                        f"{inputs_embeds[target_start:target_start+3, 0:3].cpu().tolist()}"
                                    )
                                    setattr(self, f'_debug_after_replacement_{seq_idx}_logged', True)
                            else:
                                # Multiple placeholders - use first one
                                target_offset, expected_length = placeholders[0]
                                target_start = start + target_offset
                                target_end = target_start + slice_len
                                if target_end > end:
                                    raise ValueError(
                                        f"Visual token target range [{target_start}, {target_end}) "
                                        f"is out of sequence bounds [{start}, {end})"
                                    )
                                inputs_embeds[target_start:target_end] = seq_vision_tokens
                                visual_pos_mask[target_start:target_end] = True
                                total_replaced += slice_len
                        else:
                            # No placeholders for this sequence - fallback to beginning
                            inputs_embeds[start : start + slice_len] = seq_vision_tokens
                            visual_pos_mask[start : start + slice_len] = True
                            total_replaced += slice_len
                    else:
                        # No seq_vision_placeholders provided - fallback to beginning
                        inputs_embeds[start : start + slice_len] = seq_vision_tokens
                        visual_pos_mask[start : start + slice_len] = True
                        total_replaced += slice_len
            else:
                # Fallback: original logic (only works correctly for single sequence)
                # For multi-sequence scenarios, this logic is INCORRECT because it scans
                # flattened input_ids without considering sequence boundaries.
                # In prefill-only mode, we should always have seq_image_indices and seq_vision_placeholders.
                if len(sequence_lengths) > 1:
                    raise ValueError(
                        f"Multi-sequence scenario detected ({len(sequence_lengths)} sequences) "
                        f"but seq_image_indices not provided. The fallback logic using flattened "
                        f"input_ids scanning is incorrect for multi-sequence scenarios. "
                        f"Please ensure seq_image_indices and seq_vision_placeholders are provided."
                    )
                
                # Get token IDs from config
                vision_start_token_id = getattr(self.config, "vision_start_token_id", None)
                image_token_id = getattr(self.config, "image_token_id", None)
                vision_end_token_id = getattr(self.config, "vision_end_token_id", None)
                
                if vision_start_token_id is None or image_token_id is None or vision_end_token_id is None:
                    raise ValueError(
                        "vision_start_token_id, image_token_id, and vision_end_token_id must be in config. "
                        "Please check your model configuration."
                    )
                
                # Flatten input_ids if it's in batch format
                if input_ids.dim() == 2:
                    input_ids_flat = input_ids.view(-1)
                else:
                    input_ids_flat = input_ids
                
                # In Qwen2.5-VL, the placeholder format in input_ids is:
                # <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>
                # where the number of <|image_pad|> tokens equals num_tokens per image.
                # However, vLLM's processor may have already expanded the placeholder
                # to multiple image_token_id tokens. So we need to find consecutive
                # sequences of image_token_id tokens.
                input_ids_flat_cpu = input_ids_flat.cpu()
                
                # Find all image_token_id positions
                is_image_token = input_ids_flat_cpu == image_token_id
                image_token_positions = torch.nonzero(is_image_token, as_tuple=False).squeeze(1)
                
                if len(image_token_positions) == 0:
                    raise ValueError("No image token placeholders found in input_ids")
                
                # Group consecutive image_token_id positions into sequences
                # Each sequence represents one image placeholder
                image_placeholder_ranges = []
                if len(image_token_positions) > 0:
                    start_idx = image_token_positions[0].item()
                    current_seq_start = start_idx
                    
                    for i in range(1, len(image_token_positions)):
                        current_pos = image_token_positions[i].item()
                        prev_pos = image_token_positions[i-1].item()
                        
                        if current_pos != prev_pos + 1:
                            # Gap found, end current sequence
                            seq_length = prev_pos - current_seq_start + 1
                            image_placeholder_ranges.append((current_seq_start, seq_length))
                            current_seq_start = current_pos
                    
                    # Add the last sequence
                    last_pos = image_token_positions[-1].item()
                    seq_length = last_pos - current_seq_start + 1
                    image_placeholder_ranges.append((current_seq_start, seq_length))
                
                num_placeholders = len(image_placeholder_ranges)
                num_images = len(image_embeddings)
                
                if num_placeholders != num_images:
                    raise ValueError(
                        f"Number of image placeholders ({num_placeholders}) does not match "
                        f"number of images ({num_images}). Please check input_ids and image inputs."
                    )
                
                # Verify that each placeholder range matches the corresponding image embedding size
                # Insert image embeddings at placeholder positions
                for (placeholder_start, placeholder_length), image_emb in zip(
                    image_placeholder_ranges, image_embeddings
                ):
                    num_tokens = image_emb.shape[0]
                    
                    if placeholder_length != num_tokens:
                        raise ValueError(
                            f"Placeholder length ({placeholder_length}) does not match "
                            f"image embedding size ({num_tokens}) at position {placeholder_start}"
                        )
                    
                    if placeholder_start + num_tokens > total_tokens:
                        raise ValueError(
                            f"Image tokens exceed sequence length: position {placeholder_start} + "
                            f"tokens {num_tokens} > total {total_tokens}"
                        )
                    
                    # Insert image embedding
                    image_emb = image_emb.to(inputs_embeds.device, inputs_embeds.dtype)
                    # Debug: check dimensions
                    if image_emb.shape[1] != inputs_embeds.shape[1]:
                        raise ValueError(
                            f"Image embedding dimension mismatch: image_emb.shape={image_emb.shape}, "
                            f"inputs_embeds.shape={inputs_embeds.shape} at position {placeholder_start}"
                        )
                    inputs_embeds[placeholder_start : placeholder_start + num_tokens] = image_emb
                    visual_pos_mask[placeholder_start : placeholder_start + num_tokens] = True
                    total_replaced += num_tokens

            vision_token_count = total_replaced

        if vision_token_count == 0:
            visual_pos_mask = None

        # Debug: log final state before language model
        if not hasattr(self, '_debug_qwen2_5vl_before_lm_logged'):
            print(
                f"[DEBUG qwen2.5vl.forward (fallback)] "
                f"=== Before Language Model ==="
            )
            print(
                f"[DEBUG qwen2.5vl.forward (fallback)] "
                f"vision_token_count: {vision_token_count}, "
                f"visual_pos_mask is not None: {visual_pos_mask is not None}"
            )
            if visual_pos_mask is not None:
                visual_token_indices = torch.nonzero(visual_pos_mask, as_tuple=False).squeeze(1)
                print(
                    f"[DEBUG qwen2.5vl.forward (fallback)] "
                    f"visual_pos_mask: {visual_pos_mask.sum().item()} True values out of {len(visual_pos_mask)}"
                )
                if len(visual_token_indices) > 0:
                    print(
                        f"[DEBUG qwen2.5vl.forward (fallback)] "
                        f"First 10 visual token positions: {visual_token_indices[:10].cpu().tolist()}"
                    )
            print(
                f"[DEBUG qwen2.5vl.forward (fallback)] "
                f"inputs_embeds shape: {inputs_embeds.shape}, "
                f"inputs_embeds[0:5, 0:3] (first 5 tokens, first 3 dims):\n"
                f"{inputs_embeds[0:5, 0:3].cpu().tolist()}"
            )
            self._debug_qwen2_5vl_before_lm_logged = True

        if positions is None:
            positions = torch.arange(
                inputs_embeds.size(0), device=inputs_embeds.device
            )

        if visual_pos_mask is not None and vision_token_count:
            visual_pos_mask = visual_pos_mask.to(inputs_embeds.device)
        else:
            visual_pos_mask = None

        hidden_states = self.language_model(
            input_ids=None,
            positions=positions,
            inputs_embeds=inputs_embeds,
            vision_token_count=vision_token_count,
            visual_pos_mask=visual_pos_mask,
        )

        return hidden_states
    
    def compute_logits(self, hidden_states):
        """Compute logits (delegate to language model)"""
        return self.language_model.compute_logits(hidden_states)


def load_qwen2_5_vl_model(model_path, config):
    """
    Load Qwen2_5-VL model
    
    Args:
        model_path: Model path
        config: Configuration object
    
    Returns:
        model: Qwen2_5VLForConditionalGeneration instance
    """
    hf_config = config.hf_config
    
    # Create model
    model = Qwen2_5VLForConditionalGeneration(hf_config)
    
    from nanovllm.utils.loader import load_model

    logger.debug("[load_qwen2_5_vl_model] Loading Qwen2_5-VL weights...")

    def name_mapping(weight_name: str) -> str | None:
        logger.debug(f"[name_mapping] Original weight name: {weight_name}")
        
        # Handle model.language_model.* -> language_model.*
        if weight_name.startswith("model.language_model."):
            sub_name = weight_name[len("model.language_model.") :]
            text_model_prefixes = (
                "model.",
                "embed_tokens.",
                "layers.",
                "norm.",
                "rotary_emb.",
            )
            if sub_name.startswith(text_model_prefixes):
                if sub_name.startswith("model."):
                    sub_name = sub_name[len("model.") :]
                sub_name = "language_model.model." + sub_name
            elif sub_name.startswith("lm_head."):
                sub_name = "language_model.lm_head." + sub_name[len("lm_head.") :]
            else:
                sub_name = "language_model." + sub_name
            return sub_name
        
        # Handle model.* (language model weights without "language_model" prefix)
        # This includes: model.layers.*, model.embed_tokens.*, model.norm.*, etc.
        # But exclude model.visual.* and model.language_model.* (already handled above)
        if weight_name.startswith("model.") and \
           not weight_name.startswith("model.visual.") and \
           not weight_name.startswith("model.language_model."):
            sub_name = weight_name[len("model.") :]
            mapped_name = "language_model.model." + sub_name
            return mapped_name
        
        # Handle lm_head.* -> language_model.lm_head.*
        if weight_name.startswith("lm_head."):
            sub_name = weight_name[len("lm_head.") :]
            mapped_name = "language_model.lm_head." + sub_name
            return mapped_name
        
        # Handle model.visual.* -> visual.vision.*
        if weight_name.startswith("model.visual."):
            sub_name = weight_name[len("model.visual.") :]
            mapped_name = "visual.vision." + sub_name
            return mapped_name
        
        # Handle visual.* -> visual.vision.* (direct mapping without model. prefix)
        # This is the actual format in the pretrained model files
        if weight_name.startswith("visual."):
            sub_name = weight_name[len("visual.") :]
            mapped_name = "visual.vision." + sub_name
            return mapped_name
        
        return None

    # Check for unmapped weights (only show warnings, not all details)
    try:
        from glob import glob
        from safetensors import safe_open
        import os
        
        weight_files = glob(os.path.join(model_path, "*.safetensors"))
        all_weight_names = []
        for file in weight_files:
            with safe_open(file, "pt", "cpu") as f:
                for key in f.keys():
                    all_weight_names.append(key)
        
        # Check which weights are NOT mapped (name_mapping returns None)
        unmapped_weights = []
        for weight_name in all_weight_names:
            mapped = name_mapping(weight_name)
            if mapped is None:
                unmapped_weights.append(weight_name)
        if unmapped_weights:
            logger.warning(f"[load_qwen2_5_vl_model] Found {len(unmapped_weights)} unmapped weights (will be skipped):")
            for name in sorted(unmapped_weights)[:10]:
                logger.warning(f"  {name}")
            if len(unmapped_weights) > 10:
                logger.warning(f"  ... and {len(unmapped_weights) - 10} more")
    except Exception as e:
        logger.warning(f"[load_qwen2_5_vl_model] Error checking weight names: {e}")
    
    # Load weights directly - pretrained model already has qkv.weight and qkv.bias
    # No need to merge q_proj, k_proj, v_proj since they're already merged in the checkpoint
    load_model(model, model_path, name_mapping=name_mapping)
    
    # Check if vision encoder weights are loaded correctly
    try:
        vision_params = dict(model.visual.named_parameters())
        vision_param_count = len(vision_params)
        vision_nonzero_count = sum(
            1 for param in vision_params.values() 
            if param.data.abs().sum() > 0
        )
        logger.info(f"[load_qwen2_5_vl_model] Vision encoder: {vision_param_count} parameters, {vision_nonzero_count} non-zero")
        
        # Check key vision parameters for NaN/Inf and missing weights
        key_params = [
            ("vision.patch_embed.proj.weight", True),
            ("vision.blocks.0.attn.qkv.weight", True),
            ("vision.blocks.0.attn.qkv.bias", True),
            ("vision.blocks.0.attn.proj.weight", True),
            ("vision.blocks.0.attn.proj.bias", True),
            ("vision.blocks.0.mlp.gate_proj.weight", True),
            ("vision.blocks.0.mlp.up_proj.weight", True),
            ("vision.blocks.0.mlp.down_proj.weight", True),
            ("vision.merger.ln_q.weight", True),
            ("vision.merger.mlp.0.weight", True),
            ("vision.merger.mlp.2.weight", True),
        ]
        missing_params = []
        nan_params = []
        zero_params = []
        for key, required in key_params:
            param = vision_params.get(key)
            if param is None:
                if required:
                    missing_params.append(key)
            else:
                if torch.isnan(param.data).any():
                    nan_params.append(key)
                if param.data.abs().sum() == 0:
                    zero_params.append(key)
        
        if missing_params:
            logger.error(f"[load_qwen2_5_vl_model] MISSING REQUIRED parameters: {missing_params}")
        if nan_params:
            logger.error(f"[load_qwen2_5_vl_model] Parameters with NaN: {nan_params}")
        if zero_params:
            logger.warning(f"[load_qwen2_5_vl_model] Zero parameters: {zero_params}")
        
        # Check for uninitialized vision parameters
        vision_uninitialized = [
            name for name, param in vision_params.items() 
            if param.data.abs().sum() == 0
        ]
        if vision_uninitialized:
            logger.warning(f"[load_qwen2_5_vl_model] Found {len(vision_uninitialized)} uninitialized vision parameters (first 10):")
            for name in vision_uninitialized[:10]:
                logger.warning(f"  {name}")
    except Exception as e:
        logger.error(f"[load_qwen2_5_vl_model] Error checking vision weights: {e}")
        import traceback
        traceback.print_exc()
    
    return model


__all__ = ["Qwen2_5VLForConditionalGeneration", "load_qwen2_5_vl_model"]
