"""Qwen3.5 embedding model."""
import torch
from torch import nn

from nanovllm.models.qwen3_5 import Qwen3_5TextModel
from nanovllm.layers.pooler import PoolingMethod, PoolingType, LastPool


class Qwen3_5Embedding(nn.Module):
    """Qwen3.5-based embedding model.

    Uses LAST pooling by default (extracts the last token's hidden state).
    """

    packed_modules_mapping = {
        "q_proj": ("q_proj", None),
        "k_proj": ("k_proj", None),
        "v_proj": ("v_proj", None),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
        pooling_type: PoolingType | str = PoolingType.LAST,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5TextModel(config)
        self.pooler = PoolingMethod.from_pooling_type(pooling_type)
        self.normalize = normalize

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            positions: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] or None
        Returns:
            embeddings: [batch_size, hidden_size]
        """
        hidden_states = self.model(input_ids, positions)
        # Reshape from [total_tokens, hidden_size] to [batch_size, seq_len, hidden_size]
        batch_size = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
        seq_len = input_ids.shape[-1]
        hidden_size = hidden_states.shape[-1]

        if len(hidden_states.shape) == 2:
            # Flattened format, need to reshape
            hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)

        # Pool the hidden states
        embeddings = self.pooler(hidden_states, attention_mask)

        # Normalize if requested
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings
