"""Jina Reranker M0 model implementation.

This is a wrapper around the transformers Qwen2VLForConditionalGeneration.
The model uses a score MLP on top of the last token's hidden state.

Reference: jina-reranker-m0/modeling.py
"""

import torch
from torch import nn
from typing import Optional

try:
    from transformers import Qwen2VLForConditionalGeneration
    TRANSFORMERS_QWEN2VL_AVAILABLE = True
except ImportError:
    TRANSFORMERS_QWEN2VL_AVAILABLE = False


class JinaRerankerM0(Qwen2VLForConditionalGeneration):
    """Jina Reranker M0 model.
    
    Wraps Qwen2VLForConditionalGeneration with a score MLP for reranking.
    Supports both text-only and multimodal (text + image) inputs.
    """
    
    def __init__(self, config) -> None:
        if not TRANSFORMERS_QWEN2VL_AVAILABLE:
            raise ImportError(
                "transformers Qwen2VL models are required. "
                "Please install transformers with: pip install transformers"
            )
        
        super().__init__(config)
        
        self.padding_side = "left"
        self.num_labels = 1
        
        # Replace lm_head with Identity — we only need hidden states
        self.lm_head = nn.Identity()
        
        # Get hidden_size from text_config if not at top level
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                hidden_size = getattr(text_config, "hidden_size", 1536)
            else:
                hidden_size = 1536
        
        # Score MLP: Linear → ReLU → Linear(1)
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_labels),
        )
        
        self.post_init()
        
        # Token ID appended to input for score prediction
        self.score_token_id = 100
        
        # Sigmoid normalization bias (from original Jina M0 implementation)
        self.logit_bias = 2.65
    
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for reranking.
        
        Returns:
            scores: [batch_size] relevance scores
        """
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        assert kwargs.pop("labels", None) is None, "labels should not be passed to forward()"
        
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            output_hidden_states=True,
            **kwargs
        )
        
        hidden_states = outputs.hidden_states[-1]
        
        # Extract last real token's hidden state (handles left/right padding)
        if attention_mask is not None:
            batch_size = hidden_states.shape[0]
            last_token_indices = attention_mask.sum(dim=1) - 1
            last_token_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                last_token_indices,
            ]
        else:
            last_token_hidden = hidden_states[:, -1, :]
        
        pooled_logits = self.score(last_token_hidden)
        raw_scores = pooled_logits.squeeze(-1)
        
        # Apply sigmoid normalization: sigmoid(raw - bias) → [0, 1]
        return torch.sigmoid(raw_scores.float() - self.logit_bias).to(raw_scores.dtype)
