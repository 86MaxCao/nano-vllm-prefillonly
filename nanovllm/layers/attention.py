import logging
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

logger = logging.getLogger(__name__)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        logger.debug(f"[Attention] Input - q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        logger.debug(f"[Attention] num_heads: {self.num_heads}, num_kv_heads: {self.num_kv_heads}, head_dim: {self.head_dim}")
        context = get_context()
        logger.debug(f"[Attention] context.is_prefill: {context.is_prefill}")
        k_cache, v_cache = self.k_cache, self.v_cache
        k_cache_shape = k_cache.shape if k_cache.numel() > 0 else 'empty'
        v_cache_shape = v_cache.shape if v_cache.numel() > 0 else 'empty'
        logger.debug(f"[Attention] k_cache shape: {k_cache_shape}, v_cache shape: {v_cache_shape}")
        if (k_cache.numel() and v_cache.numel() and 
            context.slot_mapping is not None and 
            context.slot_mapping.numel() > 0):
            logger.debug(f"[Attention] Storing KV cache, slot_mapping shape: {context.slot_mapping.shape}")
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            logger.debug(f"[Attention] Prefill phase")
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            logger.debug(f"[Attention] Before flash_attn_varlen_func - q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
            logger.debug(f"[Attention] max_seqlen_q: {context.max_seqlen_q}, max_seqlen_k: {context.max_seqlen_k}")
            cu_seqlens_q_shape = context.cu_seqlens_q.shape if context.cu_seqlens_q is not None else None
            logger.debug(f"[Attention] cu_seqlens_q shape: {cu_seqlens_q_shape}")
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            logger.debug(f"[Attention] Decode phase")
            # In decode phase, q shape should be [batch_size, num_heads, head_dim]
            # flash_attn_with_kvcache expects [batch_size, seqlen_q=1, num_heads, head_dim]
            # Verify q shape first
            if q.dim() != 3:
                raise ValueError(
                    f"[Attention] Decode phase: q must be 3D tensor, got {q.dim()}D with shape {q.shape}. "
                    f"num_heads={self.num_heads}, head_dim={self.head_dim}"
                )
            if q.shape[1] != self.num_heads or q.shape[2] != self.head_dim:
                raise ValueError(
                    f"[Attention] Decode phase: q shape mismatch. Expected [batch_size, num_heads={self.num_heads}, head_dim={self.head_dim}], "
                    f"got {q.shape}. This suggests q was not properly reshaped in the attention layer. "
                    f"Please check if q.view(-1, num_heads, head_dim) was called correctly."
                )
            # Use unsqueeze (same as qwen3_vl which works correctly)
            q_reshaped = q.unsqueeze(1)  # [batch_size, 1, num_heads, head_dim]
            logger.debug(f"[Attention] Before flash_attn_with_kvcache - q shape: {q.shape}, q.unsqueeze(1) shape: {q_reshaped.shape}")
            logger.debug(f"[Attention] Expected q shape: (batch_size={q.shape[0]}, seqlen_q=1, num_heads={self.num_heads}, head_dim={self.head_dim})")
            logger.debug(f"[Attention] k_cache shape: {k_cache.shape}, v_cache shape: {v_cache.shape}")
            context_lens_shape = context.context_lens.shape if (hasattr(context, 'context_lens') and context.context_lens is not None) else None
            logger.debug(f"[Attention] context.context_lens shape: {context_lens_shape}")
            # Verify q_reshaped shape before calling flash_attn_with_kvcache
            expected_shape = (q.shape[0], 1, self.num_heads, self.head_dim)
            if q_reshaped.shape != expected_shape:
                raise ValueError(
                    f"[Attention] Decode phase: q_reshaped shape mismatch. Expected {expected_shape}, got {q_reshaped.shape}"
                )
            try:
                o = flash_attn_with_kvcache(q_reshaped, k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
            except RuntimeError as e:
                # Add more context to the error message
                raise RuntimeError(
                    f"[Attention] flash_attn_with_kvcache failed. q_reshaped shape: {q_reshaped.shape}, "
                    f"expected: (batch_size, seqlen_q=1, num_heads={self.num_heads}, head_dim={self.head_dim}). "
                    f"Original error: {e}"
                ) from e
        logger.debug(f"[Attention] Output o shape: {o.shape}")
        return o
