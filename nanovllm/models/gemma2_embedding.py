"""Gemma2 embedding model for bge-multilingual-gemma2."""
import torch
from torch import nn
from transformers import Gemma2Config

from nanovllm.models.gemma2 import Gemma2Model
from nanovllm.layers.pooler import PoolingMethod, PoolingType, LastPool
from nanovllm.layers.linear import ReplicatedLinear


class Gemma2Embedding(nn.Module):
    """Gemma2-based embedding model.
    
    Uses LAST pooling by default (extracts the last token's hidden state).
    """
    
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(
        self,
        config: Gemma2Config,
        pooling_type: PoolingType | str = PoolingType.LAST,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma2Model(config)
        self.pooler = PoolingMethod.from_pooling_type(pooling_type)
        self.normalize = normalize
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_lengths: list[int] | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)

        if sequence_lengths is not None and len(hidden_states.shape) == 2:
            # Varlen mode: hidden_states is [total_tokens, hidden_size],
            # split by sequence_lengths then pad for pooling.
            splits = hidden_states.split(sequence_lengths)
            max_len = max(sequence_lengths)
            padded = torch.zeros(len(splits), max_len, hidden_states.shape[-1],
                                 dtype=hidden_states.dtype, device=hidden_states.device)
            mask = torch.zeros(len(splits), max_len,
                               dtype=torch.bool, device=hidden_states.device)
            for i, s in enumerate(splits):
                padded[i, :s.shape[0]] = s
                mask[i, :s.shape[0]] = True
            hidden_states = padded
            attention_mask = mask
        elif len(hidden_states.shape) == 2:
            batch_size = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
            seq_len = input_ids.shape[-1]
            hidden_states = hidden_states.view(batch_size, seq_len, -1)

        embeddings = self.pooler(hidden_states, attention_mask)

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings
