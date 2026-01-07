"""Encoder-only attention for bidirectional models."""
import torch
from torch import nn

from flash_attn import flash_attn_varlen_func
from nanovllm.utils.context import get_context


class EncoderOnlyAttention(nn.Module):
    """Encoder-only attention (bidirectional, no causal mask)."""
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass with bidirectional attention (causal=False)."""
        context = get_context()
        # Encoder-only attention doesn't use KV cache
        # All tokens can attend to all tokens
        o = flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=self.scale,
            causal=False,  # Bidirectional attention
        )
        return o
