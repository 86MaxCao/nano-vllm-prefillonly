"""JinaRerankerV3Model with Flash Attention (causal mask) for listwise reranking.

This implementation uses standard Flash Attention with causal mask to match
Transformers baseline behavior. For future optimization with custom listwise masks,
FlexAttention support is available in flex_attention.py.
"""
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding
from nanovllm.layers.attention import Attention


class JinaRerankerV3Attention(nn.Module):
    """Attention layer using Flash Attention with causal mask.
    
    This matches Transformers baseline behavior. Uses standard causal mask where
    each token can attend to itself and all previous tokens.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        use_flex_attention: bool = False,  # Default to False, use Flash Attention
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
        # Use standard Flash Attention (causal mask)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_mask: torch.Tensor | None = None,  # Ignored, kept for API compatibility
    ) -> torch.Tensor:
        """Forward pass using Flash Attention with causal mask.
        
        Args:
            positions: Position IDs
            hidden_states: Input hidden states
            block_mask: Ignored (kept for API compatibility)
        """
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        
        # Use standard Flash Attention (causal mask)
        o = self.attn(q, k, v)
        
        output = self.o_proj(o.flatten(1, -1))
        return output


class JinaRerankerV3DecoderLayer(nn.Module):
    """Decoder layer using Flash Attention with causal mask."""
    
    def __init__(
        self,
        config: Qwen3Config,
        use_flex_attention: bool = False,  # Default to False, use Flash Attention
    ) -> None:
        super().__init__()
        self.self_attn = JinaRerankerV3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            use_flex_attention=use_flex_attention,
        )
        # Reuse MLP from Qwen3
        from nanovllm.models.qwen3 import Qwen3MLP
        self.mlp = Qwen3MLP(
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
        block_mask: torch.Tensor | None = None,  # Ignored, kept for API compatibility
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, block_mask=block_mask)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class JinaRerankerV3Model(nn.Module):
    """Qwen3Model using Flash Attention with causal mask for listwise reranking.
    
    This matches Transformers baseline behavior using standard causal mask.
    """
    
    def __init__(
        self,
        config: Qwen3Config,
        use_flex_attention: bool = False,  # Default to False, use Flash Attention
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            JinaRerankerV3DecoderLayer(config, use_flex_attention=use_flex_attention)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_flex_attention = use_flex_attention
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_mask: torch.Tensor | None = None,  # Ignored, kept for API compatibility
    ) -> torch.Tensor:
        """Forward pass using Flash Attention with causal mask.
        
        Args:
            input_ids: Token IDs
            positions: Position IDs
            block_mask: Ignored (kept for API compatibility)
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, block_mask=block_mask)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
