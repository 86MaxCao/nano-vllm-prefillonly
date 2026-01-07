"""Gemma2 model with additional layer norms."""
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Gemma2Config

from nanovllm.models.gemma import (
    GemmaRMSNorm,
    GemmaAttention,
    GemmaMLP,
    GemmaForCausalLM,
)
from nanovllm.layers.embed_head import VocabParallelEmbedding


class Gemma2DecoderLayer(nn.Module):
    """Gemma2 decoder layer with additional feedforward layer norms.

    Gemma2 differs from Gemma by having additional layer norms:
    - pre_feedforward_layernorm (before MLP)
    - post_feedforward_layernorm (after MLP)
    """

    def __init__(
        self,
        config: Gemma2Config,
    ) -> None:
        super().__init__()
        self.self_attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, 'rope_theta', 10000),
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, 'hidden_act', None),
            hidden_activation=getattr(config, 'hidden_activation', None),
        )
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Gemma2-specific layer norms
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Input layer norm
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )

        # Self attention
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Feedforward with Gemma2-specific layer norms
        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        # post_feedforward_layernorm doesn't take residual in vLLM
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        return hidden_states, residual


class Gemma2Model(nn.Module):
    """Gemma2 model with Gemma2-specific decoder layers."""

    def __init__(
        self,
        config: Gemma2Config,
    ) -> None:
        super().__init__()
        # Validate that Gemma2 uses gelu_pytorch_tanh
        hidden_act = getattr(config, 'hidden_act', None)
        hidden_activation = getattr(config, 'hidden_activation', None)
        # Check activation: if either is set, it should be gelu_pytorch_tanh
        activation = hidden_activation or hidden_act
        if activation is not None and activation != "gelu_pytorch_tanh":
            raise ValueError(
                f"Gemma2 uses `gelu_pytorch_tanh` as the hidden activation "
                f"function, but got `{activation}`. Please set `hidden_act` "
                f"and/or `hidden_activation` to `gelu_pytorch_tanh`."
            )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList([
            Gemma2DecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gemma normalizes embeddings by sqrt(hidden_size)
        normalizer = config.hidden_size ** 0.5
        self.register_buffer(
            "normalizer", torch.tensor(normalizer), persistent=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * self.normalizer  # Gemma norm
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Gemma2ForCausalLM(GemmaForCausalLM):
    """Gemma2ForCausalLM with Gemma2Model."""

    packed_modules_mapping = GemmaForCausalLM.packed_modules_mapping

    def __init__(self, config: Gemma2Config) -> None:
        # Override model initialization to use Gemma2Model
        super(GemmaForCausalLM, self).__init__()
        self.config = config
        self.model = Gemma2Model(config)

        # For Gemma2, lm_head shares weights with embed_tokens
        # (tie_word_embeddings=True). So we reuse embed_tokens
        self.lm_head = self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # For Gemma2, lm_head is embed_tokens, so we use it directly
        return F.linear(hidden_states, self.lm_head.weight)
