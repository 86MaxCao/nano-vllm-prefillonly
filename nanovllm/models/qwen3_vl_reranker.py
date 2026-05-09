"""Qwen3-VL Reranker model implementation.

This adapter wraps Qwen3VLForConditionalGeneration for reranking tasks.
It calls the parent forward to get hidden states, then applies score_head.

Supports both:
1. Original Qwen3-VL-Reranker (uses "no"/"yes" token logits)
2. Converted sequence classification model (has score_head)
"""

import logging
from typing import Optional

import torch
from torch import nn

from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
from nanovllm.layers.linear import ReplicatedLinear

logger = logging.getLogger(__name__)


class Qwen3VLReranker(Qwen3VLForConditionalGeneration):
    """Qwen3-VL Reranker model for sequence classification/reranking tasks."""

    def __init__(
        self,
        config,
        is_original_reranker: bool = False,
        classifier_from_token: list[str] | None = None,
    ) -> None:
        super().__init__(config)

        self.is_original_reranker = is_original_reranker
        self.classifier_from_token = classifier_from_token or ["no", "yes"]
        self.padding_side = "left"
        self.num_labels = 1

        if not is_original_reranker:
            # Converted model: directly use score_head
            self.score_head = ReplicatedLinear(
                self.text_config.hidden_size,
                1,
                bias=False,
            )
        else:
            # Original reranker: create temporary lm_head for weight loading
            assert len(self.classifier_from_token) == 2, \
                "Original Qwen3-VL-Reranker requires exactly 2 tokens (e.g., ['no', 'yes'])"
            from nanovllm.layers.embed_head import ParallelLMHead
            self.language_model.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size
            )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        sequence_lengths: list[int] | None = None,
        vision_slices_per_seq: list[list[dict]] | None = None,
        seq_image_indices: list[tuple[int, int]] | None = None,
        seq_vision_placeholders: list[list[tuple[int, int]]] | None = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for reranking.

        When called without sequence_lengths (text-only varlen path from
        model_runner.rerank), returns raw hidden_states so that model_runner
        can call compute_score() separately.

        When called with sequence_lengths (multimodal or direct call), computes
        and returns scores directly.

        Returns:
            hidden_states or scores depending on whether sequence_lengths is provided
        """
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        kwargs.pop("labels", None)
        kwargs.pop("attention_mask", None)

        # Get hidden states from parent forward
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            sequence_lengths=sequence_lengths,
            vision_slices_per_seq=vision_slices_per_seq,
            seq_image_indices=seq_image_indices,
            seq_vision_placeholders=seq_vision_placeholders,
        )

        # Text-only varlen path: model_runner will call compute_score() separately
        if sequence_lengths is None:
            return hidden_states

        # Multimodal / direct call path: compute scores here
        # Compute cu_seqlens from sequence_lengths
        batch_size = len(sequence_lengths)
        cu_seqlens = torch.zeros(
            batch_size + 1, dtype=torch.int64, device=hidden_states.device
        )
        seq_lens_tensor = torch.tensor(
            sequence_lengths, dtype=torch.int64, device=hidden_states.device
        )
        torch.cumsum(seq_lens_tensor, dim=0, out=cu_seqlens[1:])

        # Extract last token hidden states
        last_token_indices = cu_seqlens[1:] - 1  # [batch_size]
        last_token_states = hidden_states[last_token_indices]  # [batch_size, hidden_size]

        # Compute scores via score_head
        if not hasattr(self, 'score_head'):
            raise RuntimeError(
                "score_head not found. For original reranker, call "
                "convert_from_original_reranker() after loading weights."
            )

        raw_scores = self.score_head(last_token_states).squeeze(-1)
        scores = torch.sigmoid(raw_scores)
        return scores

    def compute_score(
        self,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute reranking scores from hidden states (for model_runner.rerank)."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        if attention_mask is not None:
            seq_lens_actual = attention_mask.sum(dim=1)
            token_indices = (seq_lens_actual - 1).to(torch.int64)
        else:
            token_indices = torch.full(
                (batch_size,), seq_len - 1,
                dtype=torch.int64,
                device=hidden_states.device,
            )

        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        selected_states = hidden_states[batch_indices, token_indices]

        if not hasattr(self, 'score_head'):
            raise RuntimeError(
                "score_head not found. For original reranker, call "
                "convert_from_original_reranker() after loading weights."
            )

        raw_scores = self.score_head(selected_states).squeeze(-1)
        scores = torch.sigmoid(raw_scores)
        return scores

    def convert_from_original_reranker(self, tokenizer):
        """Convert original Qwen3-VL-Reranker weights to score_head.

        For models with tie_word_embeddings=True (common for Qwen3-VL-Reranker),
        the lm_head weights are not stored in the checkpoint. We compute the
        score_head directly from embed_tokens in float32 to avoid bf16 overflow.
        """
        if not self.is_original_reranker:
            return

        false_token, true_token = self.classifier_from_token
        false_id = tokenizer.convert_tokens_to_ids(false_token)
        true_id = tokenizer.convert_tokens_to_ids(true_token)

        if false_id is None or true_id is None:
            raise ValueError(
                f"Could not find token IDs for {false_token} and/or {true_token}."
            )

        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        # Determine the source for yes/no token weights
        # For tie_word_embeddings models, use embed_tokens directly
        # to avoid uninitialized lm_head weights
        tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)
        embed_tokens = getattr(self.language_model.model, 'embed_tokens', None)

        if tie_word_embeddings and embed_tokens is not None:
            # Use embed_tokens directly — always reliable
            weight_true = embed_tokens.weight.data[[true_id]].to(torch.float32)
            weight_false = embed_tokens.weight.data[[false_id]].to(torch.float32)
            score_weight = weight_true - weight_false
        elif hasattr(self.language_model, 'lm_head'):
            lm_head = self.language_model.lm_head
            import torch.distributed as dist
            tp_size = getattr(lm_head, 'tp_size', 1)

            if tp_size > 1:
                all_weights = [
                    torch.empty_like(lm_head.weight.data)
                    for _ in range(tp_size)
                ]
                dist.all_gather(all_weights, lm_head.weight.data)
                if dist.get_rank() == 0:
                    full_weight = torch.cat(all_weights, dim=0)
                    weight_true = full_weight[[true_id]].to(torch.float32)
                    weight_false = full_weight[[false_id]].to(torch.float32)
                    score_weight = weight_true - weight_false
                else:
                    score_weight = torch.zeros(
                        1, self.text_config.hidden_size,
                        dtype=torch.float32,
                        device=model_device
                    )
                dist.broadcast(score_weight, src=0)
            else:
                vocab_start_idx = getattr(lm_head, 'vocab_start_idx', 0)
                vocab_end_idx = getattr(lm_head, 'vocab_end_idx', lm_head.weight.data.shape[0])

                if (vocab_start_idx <= true_id < vocab_end_idx and
                    vocab_start_idx <= false_id < vocab_end_idx):
                    local_true_id = true_id - vocab_start_idx
                    local_false_id = false_id - vocab_start_idx
                    weight_true = lm_head.weight.data[[local_true_id]].to(torch.float32)
                    weight_false = lm_head.weight.data[[local_false_id]].to(torch.float32)
                    score_weight = weight_true - weight_false
                else:
                    weight_true = lm_head.weight.data[[true_id]].to(torch.float32)
                    weight_false = lm_head.weight.data[[false_id]].to(torch.float32)
                    score_weight = weight_true - weight_false
        else:
            raise RuntimeError(
                "Cannot convert original reranker: no embed_tokens (tied) or lm_head found."
            )

        # Clamp to prevent overflow when converting back to model dtype
        score_weight = score_weight.clamp(
            torch.finfo(model_dtype).min / 2,
            torch.finfo(model_dtype).max / 2,
        ).to(model_dtype)

        # Ensure CUDA operations complete before creating score_head
        if score_weight.is_cuda:
            torch.cuda.synchronize()

        self.score_head = ReplicatedLinear(
            self.text_config.hidden_size,
            1,
            bias=False,
        )
        self.score_head = self.score_head.to(model_device)
        # Use direct assignment instead of copy_ to avoid issues with
        # uninitialized CUDA memory (torch.set_default_device("cuda"))
        with torch.no_grad():
            self.score_head.weight.copy_(score_weight)

        if hasattr(self.language_model, 'lm_head'):
            delattr(self.language_model, 'lm_head')
        logger.info("Converted original Qwen3-VL-Reranker weights to score_head")
