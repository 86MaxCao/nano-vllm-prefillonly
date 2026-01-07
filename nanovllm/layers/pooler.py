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
            # Check if using left padding (all sequences end with padding)
            # If attention_mask[:, -1].sum() == batch_size, all last tokens
            # are padding, meaning left padding is used
            batch_size = hidden_states.shape[0]
            left_padding = (
                attention_mask[:, -1].sum() == batch_size
            )
            
            if left_padding:
                # Left padding: last token is the actual last token
                return hidden_states[:, -1]
            else:
                # Right padding: get the last non-padding token
                seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexed
                indices = torch.arange(batch_size, device=hidden_states.device)
                return hidden_states[indices, seq_lengths]
        else:
            # Simply take the last token
            return hidden_states[:, -1]


class CLSPool(PoolingMethod):
    """Pool the CLS token (first token)."""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract the first token's hidden state."""
        return hidden_states[:, 0]


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
            # Expand attention_mask to match hidden_states shape
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            # Sum over sequence dimension, weighted by attention_mask
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            return hidden_states.mean(dim=1)
