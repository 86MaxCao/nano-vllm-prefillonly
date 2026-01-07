"""Jina Reranker M0 model implementation.

This is a wrapper around the transformers JinaVLForRanking implementation.
Reference: raw_models/jina_reranker_m0/modeling.py
"""

import logging
import torch
from torch import nn
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import transformers
    from transformers import Qwen2VLForConditionalGeneration
    TRANSFORMERS_QWEN2VL_AVAILABLE = True
except ImportError:
    TRANSFORMERS_QWEN2VL_AVAILABLE = False
    logger.warning("transformers Qwen2VL models not available")


class JinaRerankerM0(Qwen2VLForConditionalGeneration):
    """Jina Reranker M0 model.
    
    This model wraps the Jina Reranker M0 architecture for reranking.
    It supports multimodal inputs (text + image) and returns relevance scores.
    
    Reference: raw_models/jina_reranker_m0/modeling.py
    
    Note: This is a wrapper around transformers implementation.
    For production use, consider implementing a native nano-vllm version.
    """
    
    def __init__(self, config) -> None:
        if not TRANSFORMERS_QWEN2VL_AVAILABLE:
            raise ImportError(
                "transformers Qwen2VL models are required. "
                "Please install transformers with: pip install transformers"
            )
        
        super().__init__(config)
        
        # Set padding side to left (important for reranking)
        self.padding_side = "left"
        self.num_labels = 1
        
        # Replace lm_head with Identity since we only want hidden states
        self.lm_head = nn.Identity()
        
        # Score MLP layer (similar to Qwen2ForRewardModel)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels),
        )
        
        # Initialize weights
        self.post_init()
        
        # Score token ID (used for appending reward token)
        self.score_token_id = 100
    
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
        
        Args:
            input_ids: [batch_size, seq_len] token ids
            positions: [batch_size, seq_len] position ids (not used, kept for compatibility)
            attention_mask: [batch_size, seq_len] or None
            pixel_values: [batch_size, num_channels, height, width] or None for images
            image_grid_thw: [batch_size, 3] or None, image grid shape (T, H, W)
            **kwargs: Other arguments passed to parent forward
        
        Returns:
            scores: [batch_size] relevance scores
        """
        # Remove unused kwargs
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        assert kwargs.pop("labels", None) is None, "labels should not be passed to forward()"
        
        # Forward through parent model
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get hidden states of the last layer
        hidden_states = outputs.hidden_states[-1]
        
        # Debug: print input info (only for first batch)
        if not hasattr(self, '_debug_logged'):
            print(f"\n[DEBUG JinaRerankerM0.forward]")
            print(f"  input_ids shape: {input_ids.shape}")
            if input_ids is not None and input_ids.numel() > 0:
                print(f"  input_ids (first 20): {input_ids[0, :20].tolist()}")
                print(f"  input_ids (last 20): {input_ids[0, -20:].tolist()}")
                if input_ids.shape[0] > 1:
                    print(f"  input_ids seq1 (first 20): {input_ids[1, :20].tolist()}")
                    print(f"  input_ids seq1 (last 20): {input_ids[1, -20:].tolist()}")
            if attention_mask is not None:
                print(f"  attention_mask shape: {attention_mask.shape}")
                print(f"  attention_mask (first 20): {attention_mask[0, :20].tolist()}")
                print(f"  attention_mask (last 20): {attention_mask[0, -20:].tolist()}")
                print(f"  attention_mask sum: {attention_mask.sum(dim=1).tolist()}")
                # Check if padding is on left or right
                first_nonzero = [
                    (attention_mask[i] != 0).nonzero(as_tuple=True)[0][0].item()
                    if (attention_mask[i] != 0).any() else -1
                    for i in range(attention_mask.shape[0])
                ]
                print(f"  first_nonzero (padding side check): {first_nonzero}")
            print(f"  hidden_states shape: {hidden_states.shape}")
            print(f"  hidden_states[:, -1, :] mean: {hidden_states[:, -1, :].mean().item():.6f}")
            print(f"  hidden_states[:, -1, :] std: {hidden_states[:, -1, :].std().item():.6f}")
            print(f"  hidden_states[:, -1, :10]: {hidden_states[0, -1, :10].tolist()}")
            if hidden_states.shape[0] > 1:
                print(f"  hidden_states seq1[:, -1, :] mean: {hidden_states[1, -1, :].mean().item():.6f}")
                print(f"  hidden_states seq1[:, -1, :] std: {hidden_states[1, -1, :].std().item():.6f}")
                print(f"  hidden_states seq1[:, -1, :10]: {hidden_states[1, -1, :10].tolist()}")
            self._debug_logged = True
        
        # IMPORTANT: padding must be on the left side
        # Get hidden states of the last token and apply score MLP
        # For left padding, the last token (index -1) is the actual last token
        # For right padding, we need to find the actual last token using attention_mask
        if attention_mask is not None:
            # Find the actual last token for each sequence
            batch_size = hidden_states.shape[0]
            last_token_indices = attention_mask.sum(dim=1) - 1  # [batch_size]
            
            # Debug: print last token info
            if not hasattr(self, '_debug_last_token_logged'):
                print(f"  last_token_indices: {last_token_indices.tolist()}")
                self._debug_last_token_logged = True
            
            # Extract hidden states at actual last token positions
            last_token_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                last_token_indices,
                :
            ]  # [batch_size, hidden_size]
            
            # Debug: print extracted last token hidden state
            if not hasattr(self, '_debug_last_hidden_logged'):
                print(f"  last_token_hidden mean: {last_token_hidden.mean().item():.6f}")
                print(f"  last_token_hidden std: {last_token_hidden.std().item():.6f}")
                print(f"  last_token_hidden[:10]: {last_token_hidden[0, :10].tolist()}")
                self._debug_last_hidden_logged = True
        else:
            # If no attention_mask, assume left padding and use last token
            last_token_hidden = hidden_states[:, -1, :]
        
        pooled_logits = self.score(last_token_hidden)
        
        # Debug: print score
        if not hasattr(self, '_debug_score_logged'):
            print(f"  pooled_logits: {pooled_logits.tolist()}")
            print(f"  final scores: {pooled_logits.squeeze(-1).tolist()}")
            self._debug_score_logged = True
        
        return pooled_logits.squeeze(-1)
    
    def set_processor(self, model_name: str):
        """Set the processor for data preprocessing."""
        from transformers import AutoProcessor
        self._processor = AutoProcessor.from_pretrained(
            model_name, max_pixels=602112, min_pixels=3136, trust_remote_code=True
        )
        return self._processor

