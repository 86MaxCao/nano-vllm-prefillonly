"""Pooling layers for embedding models."""
import torch
from torch import nn
import torch.nn.functional as F
from enum import IntEnum


class PoolingType(IntEnum):
    """Enumeration for different types of pooling methods."""
    LAST = 0
    ALL = 1
    CLS = 2
    MEAN = 4


class PoolingMethod(nn.Module):
    """Base class for pooling methods."""

    @staticmethod
    def from_pooling_type(pooling_type: PoolingType | str) -> "PoolingMethod":
        """Create a pooling method from a pooling type."""
        if isinstance(pooling_type, str):
            pooling_type = PoolingType[pooling_type.upper()]

        if pooling_type == PoolingType.LAST:
            return LastPool()
        elif pooling_type == PoolingType.CLS:
            return CLSPool()
        elif pooling_type == PoolingType.MEAN:
            return MeanPool()
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] or None
        Returns:
            pooled_states: [batch_size, hidden_size]
        """
        raise NotImplementedError

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Pool directly from varlen-packed hidden states without pad-back.

        Args:
            hidden_states: [total_tokens, hidden_size] packed without padding
            cu_seqlens: [batch_size + 1] cumulative sequence lengths
        Returns:
            pooled_states: [batch_size, hidden_size]
        """
        raise NotImplementedError


class LastPool(PoolingMethod):
    """Pool the last token's hidden state."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Extract the last token's hidden state.
        If attention_mask is provided, handles left padding correctly:
        - For left padding: returns the last token ([:, -1])
        - For right padding: returns the last non-padding token
        """
        if attention_mask is not None:
            batch_size = hidden_states.shape[0]
            left_padding = (
                attention_mask[:, -1].sum() == batch_size
            )

            if left_padding:
                return hidden_states[:, -1]
            else:
                seq_lengths = attention_mask.sum(dim=1) - 1
                indices = torch.arange(batch_size, device=hidden_states.device)
                return hidden_states[indices, seq_lengths]
        else:
            return hidden_states[:, -1]

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # Last token of each sequence is at cu_seqlens[i+1] - 1
        last_indices = cu_seqlens[1:] - 1  # [batch_size]
        return hidden_states[last_indices]


class CLSPool(PoolingMethod):
    """Pool the CLS token (first token)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract the first token's hidden state."""
        return hidden_states[:, 0]

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # First token of each sequence is at cu_seqlens[:-1]
        first_indices = cu_seqlens[:-1]  # [batch_size]
        return hidden_states[first_indices]


class MeanPool(PoolingMethod):
    """Mean pooling over all tokens."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Mean pooling over all tokens.
        If attention_mask is provided, only pool over non-padding tokens.
        """
        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        else:
            return hidden_states.mean(dim=1)

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = cu_seqlens.shape[0] - 1
        hidden_size = hidden_states.shape[-1]
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.float32)

        # Segment-reduce per sequence. A global cumsum would accumulate error
        # proportional to the whole batch rather than one sequence.
        segment_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=hidden_states.device),
            (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long),
        )
        # Accumulate in float32 like forward() does: bfloat16 index_add_
        # drifts by ~1e-3 relative error on long sequences.
        sums = torch.zeros(
            batch_size, hidden_size,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        sums.index_add_(0, segment_ids, hidden_states.float())
        return sums / seq_lens.clamp_min(1).unsqueeze(-1)
