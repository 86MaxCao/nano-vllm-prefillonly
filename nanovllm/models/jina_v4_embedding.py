"""Jina Embeddings V4 model implementation.

This is a wrapper around the transformers JinaEmbeddingsV4Model implementation.
Reference: raw_models/jina_embeddings_v4/modeling_jina_embeddings_v4.py
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union, List

import torch
from torch import nn

logger = logging.getLogger(__name__)

try:
    import transformers
    from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    TRANSFORMERS_QWEN2_5_VL_AVAILABLE = True
except ImportError:
    TRANSFORMERS_QWEN2_5_VL_AVAILABLE = False
    logger.warning("transformers Qwen2.5-VL models not available")


@dataclass
class JinaEmbeddingsV4ModelOutput:
    """Output from Jina Embeddings V4 model.
    
    Args:
        vlm_last_hidden_states: Last hidden states of the VLM (optional)
        single_vec_emb: Single-vector embeddings (optional)
        multi_vec_emb: Multi-vector embeddings (optional)
    """
    vlm_last_hidden_states: Optional[torch.Tensor] = None
    single_vec_emb: Optional[torch.Tensor] = None
    multi_vec_emb: Optional[torch.Tensor] = None


class JinaEmbeddingsV4(Qwen2_5_VLForConditionalGeneration):
    """Jina Embeddings V4 model.
    
    This model wraps the Jina Embeddings V4 architecture for embedding extraction.
    It supports multimodal inputs (text + image) and returns both single-vector
    and multi-vector embeddings.
    
    Reference: raw_models/jina_embeddings_v4/modeling_jina_embeddings_v4.py
    
    Note: This is a wrapper around transformers implementation.
    For production use, consider implementing a native nano-vllm version.
    """
    
    def __init__(
        self,
        config,
        pooling_type: str = "mean",
        normalize: bool = True,
    ) -> None:
        if not TRANSFORMERS_QWEN2_5_VL_AVAILABLE:
            raise ImportError(
                "transformers Qwen2.5-VL models are required. "
                "Please install transformers with: pip install transformers"
            )
        
        super().__init__(config)
        
        # Initialize projection layer for multi-vector embeddings
        self._init_projection_layer(config)
        
        # Store pooling and normalization settings
        self.pooling_type = pooling_type
        self.normalize = normalize
        
        # Initialize weights
        self.post_init()
    
    def _init_projection_layer(self, config) -> None:
        """Initialize projection layers for multi-vector embeddings."""
        multi_vector_projector_dim = getattr(config, 'multi_vector_projector_dim', 128)
        
        self.multi_vector_projector = nn.Linear(
            in_features=self.config.text_config.hidden_size,
            out_features=multi_vector_projector_dim,
        )
    
    def get_last_hidden_states(
        self,
        task_label: Union[str, List[str]],
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Get last hidden states from the model."""
        if "pixel_values" in kwargs:
            pixel_values = kwargs["pixel_values"]
            # Check if pixel_values is already a concatenated tensor
            # (from llm_engine.py batch processing)
            if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 2:
                # Already in correct format (total_patches, cps), no processing needed
                # The pixel_values is already concatenated from all images in the batch
                pass
            elif isinstance(pixel_values, (list, tuple)):
                # If it's a list/tuple, process each element separately
                # This is the original logic for handling per-image pixel_values
                offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
                kwargs["pixel_values"] = torch.cat(
                    [pv[:o] for pv, o in zip(pixel_values, offsets)], dim=0
                )
            else:
                # Other cases: assume it's already in the correct format
                # or needs to be passed through as-is
                pass
        
        # Get position_ids and rope_deltas
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=kwargs.get("image_grid_thw", None),
            attention_mask=attention_mask,
        )
        
        kwargs["output_hidden_states"] = True
        outputs = super().forward(
            task_label=task_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            use_cache=False,
        )
        
        hidden_states = outputs.hidden_states
        if not hidden_states:
            raise ValueError("Hidden states not found in model output")
        
        return hidden_states[-1]
    
    def get_single_vector_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Get single-vector embeddings from hidden states.
        
        Uses mean pooling, with special handling for images.
        """
        if input_ids is not None and self._input_has_image(input_ids[0]):
            # Image input: pool only over image tokens
            img_start_positions = torch.where(
                input_ids == self.config.vision_start_token_id
            )[1]
            img_end_positions = torch.where(
                input_ids == self.config.vision_end_token_id
            )[1]
            
            batch_size, seq_len = input_ids.shape
            position_indices = torch.arange(seq_len, device=input_ids.device).expand(
                batch_size, -1
            )
            image_mask = (position_indices >= img_start_positions.unsqueeze(1)) & (
                position_indices <= img_end_positions.unsqueeze(1)
            )
            
            masked_hidden_states = hidden_states * image_mask.unsqueeze(-1)
            pooled_output = masked_hidden_states.sum(dim=1) / image_mask.sum(
                dim=1, keepdim=True
            )
        else:
            # Text input: mean pooling over all tokens
            pooled_output = torch.sum(
                hidden_states * attention_mask.unsqueeze(-1), dim=1
            ) / torch.sum(attention_mask, dim=1, keepdim=True)
        
        return torch.nn.functional.normalize(pooled_output, dim=-1)
    
    def get_multi_vector_embeddings(
        self,
        task_label: Union[str, List[str]],
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project hidden states to multi-vector embeddings."""
        # Simple linear projection (in reference implementation, this uses MultiAdapterLinear with task_label)
        # For now, we use a simple linear layer
        multi_vec_emb = self.multi_vector_projector(hidden_states)
        multi_vec_emb = torch.nn.functional.normalize(multi_vec_emb, dim=-1)
        return multi_vec_emb * attention_mask.unsqueeze(-1)
    
    def _input_has_image(self, input_ids):
        """Check if input contains image tokens."""
        return self.config.vision_start_token_id in input_ids
    
    def forward(
        self,
        task_label: Union[str, List[str]] = "retrieval",
        input_ids: torch.LongTensor | None = None,
        positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        output_vlm_last_hidden_states: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for embedding extraction.
        
        Args:
            task_label: Task label (e.g., "retrieval", "text-matching", "code")
            input_ids: [batch_size, seq_len] token ids
            positions: [batch_size, seq_len] position ids (not used, kept for compatibility)
            attention_mask: [batch_size, seq_len] or None
            pixel_values: [batch_size, num_channels, height, width] or None for images
            image_grid_thw: [batch_size, 3] or None, image grid shape (T, H, W)
            output_vlm_last_hidden_states: Whether to return hidden states
            **kwargs: Other arguments
        
        Returns:
            embeddings: [batch_size, hidden_size] single-vector embeddings
        """
        # Get hidden states
        hidden_states = self.get_last_hidden_states(
            task_label=task_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs
        )
        
        # Compute single-vector embeddings (we return this as the main output)
        single_vec_emb = self.get_single_vector_embeddings(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        
        # For now, we only return single-vector embeddings
        # Multi-vector embeddings can be added if needed
        return single_vec_emb
    
    def set_processor(self, model_name: str):
        """Set the processor for data preprocessing."""
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
        return self.processor

