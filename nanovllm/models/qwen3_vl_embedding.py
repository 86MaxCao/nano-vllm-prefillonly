"""Qwen3-VL Embedding model implementation.

This adapter wraps Qwen3VLForConditionalGeneration for embedding extraction.
It calls the parent forward to get hidden states, then applies pooling.
"""

import logging
from typing import Optional

import torch
from torch import nn

from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
from nanovllm.layers.pooler import PoolingMethod

logger = logging.getLogger(__name__)


class Qwen3VLEmbedding(Qwen3VLForConditionalGeneration):
    """Qwen3-VL Embedding model.

    Inherits from Qwen3VLForConditionalGeneration and overrides forward
    to apply pooling + normalization on hidden states for embedding extraction.
    """

    def __init__(
        self,
        config,
        pooling_type: str = "LAST",
        normalize: bool = True,
    ) -> None:
        super().__init__(config)

        self.pooler = PoolingMethod.from_pooling_type(pooling_type)
        self.normalize = normalize

        # Expose language_model.model as .model for text-only embedding path
        # This allows model_runner to find embed_tokens via self.model
        self.model = self.language_model.model

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
        """Forward pass for embedding extraction.

        Returns:
            embeddings: [batch_size, hidden_size] normalized embeddings
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

        if sequence_lengths is None:
            raise ValueError("sequence_lengths must be provided for embedding extraction")

        # Compute cu_seqlens from sequence_lengths
        batch_size = len(sequence_lengths)
        cu_seqlens = torch.zeros(
            batch_size + 1, dtype=torch.int64, device=hidden_states.device
        )
        seq_lens_tensor = torch.tensor(
            sequence_lengths, dtype=torch.int64, device=hidden_states.device
        )
        torch.cumsum(seq_lens_tensor, dim=0, out=cu_seqlens[1:])

        # Extract last token embeddings using cu_seqlens
        last_token_indices = cu_seqlens[1:] - 1  # [batch_size]
        embeddings = hidden_states[last_token_indices]  # [batch_size, hidden_size]

        # Normalize
        embeddings = embeddings.to(torch.float32)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings
