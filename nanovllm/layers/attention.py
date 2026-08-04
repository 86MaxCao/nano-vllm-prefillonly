import functools
import logging

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _paged_kvcache_is_usable() -> bool:
    """Check whether ``flash_attn_with_kvcache`` handles a paged cache.

    flash-attn 2.8.4 requires a paged block size divisible by 256 yet returns
    NaN for any cache whose page dimension exceeds 128, so paged decode is
    unusable in that build. Probing once lets the decode path raise instead of
    emitting NaN logits that sample into plausible-looking tokens.
    """
    if not torch.cuda.is_available():
        return False
    try:
        head_dim = 64
        q = torch.zeros(1, 1, 2, head_dim, dtype=torch.float16, device="cuda")
        q[..., 0] = 1.0
        cache = torch.zeros(1, 256, 1, head_dim, dtype=torch.float16, device="cuda")
        cache[0, 0, 0, 0] = 1.0
        out = flash_attn_with_kvcache(
            q,
            cache,
            cache,
            cache_seqlens=torch.ones(1, dtype=torch.int32, device="cuda"),
            block_table=torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
            softmax_scale=1.0,
            causal=True,
        )
        return bool(torch.isfinite(out).all())
    except Exception as exc:  # noqa: BLE001 - any failure means "unusable"
        logger.debug(
            "flash_attn_with_kvcache probe raised %s: %s", type(exc).__name__, exc
        )
        return False


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
        softcap: float = 0.0,
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # Gemma2 caps pre-softmax logits and alternates local/global attention;
        # 0.0 and None disable each feature for every other architecture.
        self.softcap = softcap or 0.0
        self.sliding_window = sliding_window
        self.k_cache = self.v_cache = torch.tensor([])

    @property
    def _window_size(self) -> tuple[int, int]:
        """Left/right context limits in flash-attn's convention (-1 = unbounded)."""
        if self.sliding_window is None:
            return (-1, -1)
        return (self.sliding_window - 1, 0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if (k_cache.numel() and v_cache.numel() and
            context.slot_mapping is not None and
            context.slot_mapping.numel() > 0):
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables,
                                       softcap=self.softcap, window_size=self._window_size)
        else:    # decode
            if not _paged_kvcache_is_usable():
                raise RuntimeError(
                    "Multi-token decoding is unavailable: this flash-attn build "
                    "returns non-finite values for paged KV caches "
                    "(flash_attn_with_kvcache requires a block size divisible by "
                    "256 but produces NaN for pages larger than 128). This engine "
                    "targets prefill-only workloads (embedding, reranking, "
                    "single-token generation); use max_tokens=1, or install a "
                    "flash-attn build with working paged decode."
                )
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True,
                                        softcap=self.softcap, window_size=self._window_size)
        return o
