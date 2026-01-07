"""Simplified implementation of the LLaVANext multimodal model.

This module implements LLaVANext following the structure of qwen3_vl.py.
Reference: transformers/src/transformers/models/llava_next/modeling_llava_next.py

Note: LLaVANext uses Mistral as the language model. Since nano-vllm may not have
a native Mistral implementation, this implementation focuses on the multimodal
projector and wrapper. The vision encoder (CLIP) and language model (Mistral)
may need to use transformers wrappers or native implementations if available.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)


# ---------------------------------------------------------------------------
# Multimodal Projector
# ---------------------------------------------------------------------------

class LlavaNextMultiModalProjector(nn.Module):
    """Multimodal projector that projects vision features to text space.
    
    Reference: transformers/src/transformers/models/llava_next/modeling_llava_next.py
    """
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_act: str = "gelu",
        multimodal_projector_bias: bool = True,
        num_feature_layers: int = 1,
    ):
        super().__init__()
        # We have hidden_size * the number of vision feature layers
        self.linear_1 = nn.Linear(
            vision_hidden_size * num_feature_layers,
            text_hidden_size,
            bias=multimodal_projector_bias,
        )
        # Activation function
        if projector_hidden_act == "gelu":
            self.act = F.gelu
        elif projector_hidden_act == "silu":
            self.act = F.silu
        elif projector_hidden_act == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"Unknown activation: {projector_hidden_act}")
        
        self.linear_2 = nn.Linear(
            text_hidden_size, text_hidden_size, bias=multimodal_projector_bias
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Multimodal Wrapper
# ---------------------------------------------------------------------------

class LlavaNextForConditionalGeneration(nn.Module):
    """LLaVANext conditional generation model.
    
    Note: This is a simplified implementation. The vision encoder (CLIP) and
    language model (Mistral) components may need to be wrapped from transformers
    or implemented natively if Mistral support is added to nano-vllm.
    
    Reference: transformers/src/transformers/models/llava_next/modeling_llava_next.py
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_config = getattr(config, "text_config", config)
        self.vision_config = getattr(config, "vision_config", None)

        if self.vision_config is None:
            raise ValueError("vision_config is missing; cannot build a multimodal model")

        # Vision encoder (CLIP) - typically handled by processor, but we need it here
        # For now, we'll use AutoModel wrapper from transformers
        try:
            from transformers import AutoModel
            self.vision_tower = AutoModel.from_config(self.vision_config)
        except ImportError:
            raise ImportError("transformers is required for vision encoder. Install with: pip install transformers")

        # Multimodal projector
        num_feature_layers = 1 if isinstance(getattr(config, "vision_feature_layer", -2), int) else len(config.vision_feature_layer)
        self.multi_modal_projector = LlavaNextMultiModalProjector(
            vision_hidden_size=self.vision_config.hidden_size,
            text_hidden_size=self.text_config.hidden_size,
            projector_hidden_act=getattr(config, "projector_hidden_act", "gelu"),
            multimodal_projector_bias=getattr(config, "multimodal_projector_bias", True),
            num_feature_layers=num_feature_layers,
        )

        # Image newline parameter
        embed_std = 1 / math.sqrt(self.text_config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(self.text_config.hidden_size) * embed_std)

        # Language model (Mistral) - for now use transformers wrapper
        # TODO: Implement native Mistral support in nano-vllm
        try:
            from transformers import AutoModel
            self.language_model = AutoModel.from_config(self.text_config)
            self.lm_head = nn.Linear(self.text_config.hidden_size, self.text_config.vocab_size, bias=False)
        except ImportError:
            raise ImportError("transformers is required for language model. Install with: pip install transformers")

        print("[LlavaNextForConditionalGeneration] Initialization complete")
        print(f"  - Vision encoder: {type(self.vision_tower).__name__}")
        print(f"  - Language model: {type(self.language_model).__name__}")
        print(f"  - Multimodal projector: {type(self.multi_modal_projector).__name__}")
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        image_sizes: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for LLaVANext.
        
        Note: This is a simplified implementation. For full functionality,
        implement get_image_features, pack_image_features, and proper
        image feature processing similar to transformers implementation.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # Get batch size and sequence length
        if input_ids is not None:
            if input_ids.dim() == 2:
                batch_size, seq_len = input_ids.shape
            else:
                batch_size = 1
                seq_len = input_ids.shape[0]
        else:
            if inputs_embeds.dim() == 3:
                batch_size, seq_len, _ = inputs_embeds.shape
            else:
                batch_size = 1
                seq_len = inputs_embeds.shape[0]
        
        # Process vision inputs if provided
        if pixel_values is not None:
            # Handle different pixel_values formats
            # LLaVA-Next processor may return:
            # 1. A list of tensors (one per image)
            # 2. A tensor with shape (batch, num_patches, c, h, w) for multi-resolution
            # 3. A tensor with shape (batch, c, h, w) for single resolution
            
            pixel_values_to_process = None
            
            if isinstance(pixel_values, list):
                # If it's a list, each element corresponds to one sequence
                # For multiple sequences, we need to process each separately
                # and insert image features into the corresponding sequence's inputs_embeds
                if len(pixel_values) > 0:
                    # Check the format of the first item
                    first_item = pixel_values[0]
                    
                    # Handle nested list (batch of lists of patches)
                    if isinstance(first_item, list):
                        # This is list[list[tensor]] format from processor
                        # Flatten: concatenate all patches from all images
                        all_patches = []
                        for image_patches in pixel_values:
                            if isinstance(image_patches, list):
                                # Each image_patches is a list of patch tensors
                                for patch in image_patches:
                                    if isinstance(patch, torch.Tensor):
                                        all_patches.append(patch)
                                    else:
                                        # Convert numpy array to tensor if needed
                                        if hasattr(patch, 'shape'):
                                            all_patches.append(torch.from_numpy(patch) if isinstance(patch, np.ndarray) else torch.tensor(patch))
                            elif isinstance(image_patches, torch.Tensor):
                                # Single tensor per image (already stacked patches)
                                all_patches.append(image_patches)
                        
                        if len(all_patches) > 0:
                            # Stack or concatenate all patches
                            if all_patches[0].dim() == 3:
                                # Each patch is (c, h, w), stack them
                                pixel_values_to_process = torch.stack(all_patches, dim=0)
                            elif all_patches[0].dim() == 4:
                                # Each item is (num_patches, c, h, w), concatenate
                                pixel_values_to_process = torch.cat(all_patches, dim=0)
                            else:
                                raise ValueError(
                                    f"Unexpected patch tensor dimensions: {all_patches[0].dim()}. "
                                    f"Shape: {all_patches[0].shape}"
                                )
                        else:
                            pixel_values_to_process = None
                    elif isinstance(first_item, torch.Tensor):
                        # Each item is a tensor for one sequence
                        # For multiple sequences, concatenate all pixel_values
                        # This will be processed as a batch, but we need to handle
                        # image feature insertion per sequence
                        if first_item.dim() == 4:
                            # Each item is (num_patches, c, h, w) - concatenate all
                            pixel_values_to_process = torch.cat(pixel_values, dim=0)
                        elif first_item.dim() == 3:
                            # Each item is (c, h, w) - single image per batch item
                            pixel_values_to_process = torch.stack(pixel_values, dim=0)
                        else:
                            raise ValueError(
                                f"Unexpected pixel_values list item dimensions: {first_item.dim()}. "
                                f"Shape: {first_item.shape}"
                            )
                    else:
                        raise ValueError(f"Unexpected pixel_values list item type: {type(first_item)}")
                else:
                    # Empty list, skip vision processing
                    pixel_values_to_process = None
            elif isinstance(pixel_values, torch.Tensor):
                # Handle tensor format
                if pixel_values.dim() == 5:
                    # Shape: (batch, num_patches, c, h, w) - multi-resolution
                    b, num_patches, c, h, w = pixel_values.shape
                    # Reshape to (batch * num_patches, c, h, w) for vision tower
                    pixel_values_to_process = pixel_values.view(b * num_patches, c, h, w)
                elif pixel_values.dim() == 4:
                    # Shape: (batch, c, h, w) - standard format
                    pixel_values_to_process = pixel_values
                elif pixel_values.dim() == 1:
                    # 1D tensor - this should NOT happen with proper processor output
                    # According to transformers, pixel_values should be a list of lists
                    # If we get 1D tensor, it means processor output was incorrectly converted
                    total_elements = pixel_values.numel()
                    
                    raise ValueError(
                        f"Received 1D pixel_values tensor with shape {pixel_values.shape} "
                        f"(total_elements: {total_elements}). This is incorrect - pixel_values should be "
                        f"a list of lists (batch of lists of patches) or a 4D/5D tensor.\n"
                        f"\n"
                        f"This usually happens when processor output is incorrectly converted. "
                        f"LLaVA-Next processor returns pixel_values as a list where each element is "
                        f"a numpy array of shape (num_patches, c, h, w).\n"
                        f"\n"
                        f"Please check:\n"
                        f"1. How pixel_values is extracted from processor output\n"
                        f"2. Whether return_tensors='pt' is causing incorrect conversion\n"
                        f"3. Try accessing pixel_values before .to('cuda') to see its original format\n"
                        f"\n"
                        f"Expected formats:\n"
                        f"- List of lists: [[patch1, patch2, ...], [patch1, patch2, ...], ...]\n"
                        f"- 5D tensor: (batch, num_patches, c, h, w)\n"
                        f"- 4D tensor: (total_patches, c, h, w)\n"
                        f"\n"
                        f"Current pixel_values type: {type(pixel_values)}, shape: {pixel_values.shape}"
                    )
                elif pixel_values.dim() == 2 or pixel_values.dim() == 3:
                    # 2D or 3D tensor - might need reshaping
                    # This could be (batch, flattened) or (c, h, w)
                    raise ValueError(
                        f"Unexpected pixel_values dimensions: {pixel_values.dim()}. "
                        f"Shape: {pixel_values.shape}. "
                        f"Expected 4D (batch, c, h, w) or 5D (batch, num_patches, c, h, w) tensor, "
                        f"or a list of tensors. If this is a flattened tensor, please reshape it first."
                    )
                else:
                    raise ValueError(
                        f"Unexpected pixel_values tensor dimensions: {pixel_values.dim()}. "
                        f"Shape: {pixel_values.shape}. Expected 4D or 5D tensor."
                    )
            else:
                raise ValueError(f"Unexpected pixel_values type: {type(pixel_values)}")
            
            # Store pixel_values_to_process for later use in per-sequence processing
            # We'll process image features per sequence if we have sequence_lengths
            pixel_values_to_process_stored = None
            if pixel_values_to_process is not None:
                # Ensure pixel_values_to_process has correct shape: (N, c, h, w)
                if pixel_values_to_process.dim() != 4:
                    raise ValueError(
                        f"pixel_values_to_process must be 4D (N, c, h, w), "
                        f"got shape: {pixel_values_to_process.shape}"
                    )
                pixel_values_to_process_stored = pixel_values_to_process
                
            # Process image features - either batch or per-sequence
            image_features = None
            if pixel_values_to_process_stored is not None:
                # Get vision features from vision tower
                vision_outputs = self.vision_tower(pixel_values_to_process_stored)
                
                # Extract features from specified layer
                vision_feature_layer = getattr(self.config, "vision_feature_layer", -2)
                if isinstance(vision_feature_layer, int):
                    if hasattr(vision_outputs, "last_hidden_state"):
                        image_features = vision_outputs.last_hidden_state
                    elif isinstance(vision_outputs, tuple):
                        image_features = vision_outputs[0]
                    else:
                        image_features = vision_outputs
                else:
                    # Multiple layers - concatenate features
                    if hasattr(vision_outputs, "hidden_states"):
                        image_features = torch.cat([vision_outputs.hidden_states[i] for i in vision_feature_layer], dim=-1)
                    else:
                        raise ValueError("Multiple vision feature layers requested but hidden_states not available")
                
                # Project vision features to text space
                image_features = self.multi_modal_projector(image_features)
                # image_features shape might be (num_patches, text_hidden_size) or (1, num_patches, text_hidden_size)
                # or (batch, num_patches, text_hidden_size) for batch processing
                # Ensure it's 2D: (num_patches, text_hidden_size)
                if image_features.dim() > 2:
                    # Flatten all dimensions except the last one
                    # e.g., (1, num_patches, hidden_size) -> (num_patches, hidden_size)
                    # or (batch, num_patches, hidden_size) -> (batch * num_patches, hidden_size)
                    shape = image_features.shape
                    hidden_size = shape[-1]
                    num_elements = shape[:-1].numel()  # Product of all dims except last
                    image_features = image_features.view(num_elements, hidden_size)
                elif image_features.dim() == 1:
                    # This shouldn't happen, but handle it
                    image_features = image_features.unsqueeze(0)
                # Now image_features should be 2D: (num_patches, text_hidden_size)
                
                # Insert image features into inputs_embeds at image placeholder positions
                # LLaVA-Next uses image_token_id to mark image placeholder positions
                # CRITICAL: inputs_embeds length must ALWAYS equal input_ids length
                if input_ids is not None:
                    image_token_id = getattr(self.config, "image_token_id", None)
                    if image_token_id is None:
                        # Try to get from text_config
                        image_token_id = getattr(self.text_config, "image_token_id", None)
                    
                    hidden_size = inputs_embeds.shape[-1]
                    original_inputs_embeds_len = inputs_embeds.shape[0]
                    original_input_ids_len = input_ids.shape[0]
                    
                    # Debug log
                    if not hasattr(self, '_debug_image_insert_logged'):
                        print("\n[DEBUG llavanext.forward] Image feature insertion:")
                        print(f"  input_ids shape: {input_ids.shape}, "
                              f"len: {original_input_ids_len}")
                        print(f"  inputs_embeds shape: {inputs_embeds.shape}, "
                              f"len: {original_inputs_embeds_len}")
                        print(f"  image_features shape: {image_features.shape}")
                        print(f"  image_token_id: {image_token_id}")
                        print(f"  sequence_lengths: {sequence_lengths}")
                        print(f"  pixel_values type: {type(pixel_values)}")
                        if isinstance(pixel_values, list):
                            print(f"  pixel_values list length: {len(pixel_values)}")
                        self._debug_image_insert_logged = True
                    
                    # Ensure inputs_embeds and input_ids have same length
                    if original_inputs_embeds_len != original_input_ids_len:
                        raise ValueError(
                            f"inputs_embeds length ({original_inputs_embeds_len}) must equal "
                            f"input_ids length ({original_input_ids_len})"
                        )
                    
                    if image_token_id is not None:
                        # Handle varlen format (flattened) with sequence_lengths
                        is_multi_seq = (
                            sequence_lengths is not None
                            and len(sequence_lengths) > 1
                            and isinstance(pixel_values, list)
                            and len(pixel_values) == len(sequence_lengths)
                        )
                        if is_multi_seq:
                            # Multiple sequences in varlen format - process each separately
                            input_ids_flat = input_ids
                            inputs_embeds_flat = inputs_embeds
                            batch_size = len(sequence_lengths)
                            
                            # Calculate cumulative sequence lengths
                            cu_seqlens = [0]
                            for seq_len in sequence_lengths:
                                cu_seqlens.append(cu_seqlens[-1] + seq_len)
                            
                            # Process each sequence's pixel_values separately
                            for seq_idx in range(batch_size):
                                seq_start = cu_seqlens[seq_idx]
                                seq_end = cu_seqlens[seq_idx + 1]
                                seq_input_ids = input_ids_flat[seq_start:seq_end]
                                seq_inputs_embeds = inputs_embeds_flat[seq_start:seq_end]
                                
                                # Process this sequence's pixel_values to get image features
                                seq_pixel_values = pixel_values[seq_idx]
                                if seq_pixel_values is not None:
                                    # Process single sequence's pixel_values
                                    if isinstance(seq_pixel_values, torch.Tensor):
                                        if seq_pixel_values.dim() == 4:
                                            # Shape: (num_patches, c, h, w)
                                            seq_pixel_values_to_process = seq_pixel_values
                                        elif seq_pixel_values.dim() == 5:
                                            # Shape: (1, num_patches, c, h, w)
                                            b, num_patches, c, h, w = seq_pixel_values.shape
                                            seq_pixel_values_to_process = seq_pixel_values.view(b * num_patches, c, h, w)
                                        else:
                                            raise ValueError(
                                                f"Unexpected seq_pixel_values dim: "
                                                f"{seq_pixel_values.dim()}"
                                            )
                                        
                                        # Get vision features for this sequence
                                        seq_vision_outputs = self.vision_tower(
                                            seq_pixel_values_to_process
                                        )
                                        
                                        # Extract features
                                        vision_feature_layer = getattr(
                                            self.config, "vision_feature_layer", -2
                                        )
                                        if isinstance(vision_feature_layer, int):
                                            if hasattr(seq_vision_outputs, "last_hidden_state"):
                                                seq_image_features = seq_vision_outputs.last_hidden_state
                                            elif isinstance(seq_vision_outputs, tuple):
                                                seq_image_features = seq_vision_outputs[0]
                                            else:
                                                seq_image_features = seq_vision_outputs
                                        else:
                                            if hasattr(seq_vision_outputs, "hidden_states"):
                                                seq_image_features = torch.cat(
                                                    [seq_vision_outputs.hidden_states[i]
                                                     for i in vision_feature_layer], dim=-1
                                                )
                                            else:
                                                raise ValueError(
                                                    "Multiple vision feature layers requested "
                                                    "but hidden_states not available"
                                                )
                                        
                                        # Project vision features to text space
                                        seq_image_features = self.multi_modal_projector(
                                            seq_image_features
                                        )
                                        
                                        # Ensure 2D
                                        if seq_image_features.dim() > 2:
                                            shape = seq_image_features.shape
                                            hidden_size_seq = shape[-1]
                                            num_elements = shape[:-1].numel()
                                            seq_image_features = seq_image_features.view(
                                                num_elements, hidden_size_seq
                                            )
                                        elif seq_image_features.dim() == 1:
                                            seq_image_features = seq_image_features.unsqueeze(0)
                                        
                                        # Find image tokens in this sequence
                                        seq_is_image_token = seq_input_ids == image_token_id
                                        seq_image_token_positions = torch.nonzero(
                                            seq_is_image_token, as_tuple=False
                                        )
                                        
                                        if len(seq_image_token_positions) > 0:
                                            seq_image_token_positions = seq_image_token_positions.squeeze(1)
                                            num_seq_image_tokens = len(seq_image_token_positions)
                                            num_seq_image_features = seq_image_features.shape[0]
                                            
                                            # Debug log for first sequence
                                            if seq_idx == 0 and not hasattr(self, '_debug_per_seq_logged'):
                                                print(f"\n[DEBUG llavanext.forward] Per-sequence processing:")
                                                print(f"  Sequence {seq_idx}:")
                                                print(f"    Range: [{seq_start}, {seq_end})")
                                                print(f"    Image tokens: {num_seq_image_tokens}")
                                                print(f"    Image features: {num_seq_image_features}")
                                                self._debug_per_seq_logged = True
                                            
                                            # Ensure device/dtype match
                                            seq_image_features = seq_image_features.to(
                                                device=seq_inputs_embeds.device,
                                                dtype=seq_inputs_embeds.dtype
                                            )
                                            
                                            # Insert image features at image token positions
                                            # Adjust positions to global coordinates
                                            global_image_token_positions = (
                                                seq_image_token_positions + seq_start
                                            )
                                            
                                            if num_seq_image_tokens == num_seq_image_features:
                                                inputs_embeds_flat[global_image_token_positions] = (
                                                    seq_image_features
                                                )
                                            elif num_seq_image_tokens > num_seq_image_features:
                                                repeat_factor = (
                                                    (num_seq_image_tokens + num_seq_image_features - 1)
                                                    // num_seq_image_features
                                                )
                                                seq_features_repeated = seq_image_features.repeat(
                                                    repeat_factor, 1
                                                )[:num_seq_image_tokens]
                                                inputs_embeds_flat[global_image_token_positions] = (
                                                    seq_features_repeated
                                                )
                                            else:
                                                seq_features_truncated = seq_image_features[
                                                    :num_seq_image_tokens
                                                ]
                                                inputs_embeds_flat[global_image_token_positions] = (
                                                    seq_features_truncated
                                                )
                            
                            # Update inputs_embeds
                            inputs_embeds = inputs_embeds_flat
                            # Skip batch-level processing since we already processed per-sequence
                            # Set a flag to skip the batch processing below
                            processed_per_sequence = True
                        else:
                            processed_per_sequence = False
                        
                        # Only do batch-level processing if not already processed per-sequence
                        if not processed_per_sequence and image_features is not None:
                            if input_ids.dim() == 2:
                                # Batch processing - flatten for processing
                                batch_size, seq_len = input_ids.shape
                                input_ids_flat = input_ids.view(-1)
                                inputs_embeds_flat = inputs_embeds.view(-1, hidden_size)
                            else:
                                # Single sequence or varlen format
                                input_ids_flat = input_ids
                                inputs_embeds_flat = inputs_embeds
                            
                            # Find image token positions
                            is_image_token = input_ids_flat == image_token_id
                            image_token_positions = torch.nonzero(is_image_token, as_tuple=False)
                            
                            if len(image_token_positions) > 0:
                                image_token_positions = image_token_positions.squeeze(1)
                                
                                # Calculate how many image tokens we have
                                num_image_tokens = len(image_token_positions)
                                num_image_features = image_features.shape[0]
                                
                                # Debug log
                                if not hasattr(self, '_debug_image_tokens_logged'):
                                    print(f"  Found {num_image_tokens} image tokens, "
                                          f"{num_image_features} image features")
                                    if num_image_tokens > 10:
                                        pos_list = image_token_positions[:10].tolist()
                                    else:
                                        pos_list = image_token_positions.tolist()
                                    print(f"  Image token positions (first 10): {pos_list}")
                                    self._debug_image_tokens_logged = True
                                
                                # Ensure image_features are on the same device and dtype
                                image_features = image_features.to(
                                    device=inputs_embeds_flat.device,
                                    dtype=inputs_embeds_flat.dtype
                                )
                                
                                # CRITICAL: Only replace, never concatenate or change length
                                if num_image_tokens == num_image_features:
                                    # Perfect match - replace directly
                                    inputs_embeds_flat[image_token_positions] = image_features
                                elif num_image_tokens > num_image_features:
                                    # More tokens than features - repeat features
                                    repeat_factor = (
                                        (num_image_tokens + num_image_features - 1)
                                        // num_image_features
                                    )
                                    image_features_repeated = image_features.repeat(
                                        repeat_factor, 1
                                    )[:num_image_tokens]
                                    inputs_embeds_flat[image_token_positions] = (
                                        image_features_repeated
                                    )
                                else:
                                    # More features than tokens - truncate to match
                                    image_features_truncated = image_features[
                                        :num_image_tokens
                                    ]
                                    inputs_embeds_flat[image_token_positions] = (
                                        image_features_truncated
                                    )
                                
                                # Update inputs_embeds
                                inputs_embeds = inputs_embeds_flat
                            else:
                                # No image tokens found - this is an error
                                print("[WARNING llavanext.forward] No image tokens found!")
                                print(f"  input_ids shape: {input_ids.shape}")
                                print(f"  image_token_id: {image_token_id}")
                                print("  This might indicate a problem with tokenization")
                                # Do NOT modify inputs_embeds length - just skip
                                pass
                    else:
                        # No image_token_id in config
                        print("[WARNING llavanext.forward] No image_token_id found!")
                        print("  Cannot insert image features without image_token_id")
                        # Do NOT modify inputs_embeds length
                        pass
                else:
                    # No input_ids - cannot insert image features
                    print("[WARNING llavanext.forward] No input_ids provided!")
                    print("  Cannot insert image features without input_ids")
                    # Do NOT modify inputs_embeds length
                    pass
                
                # Final check: ensure inputs_embeds length matches input_ids length
                if input_ids is not None:
                    final_inputs_embeds_len = inputs_embeds.shape[0]
                    final_input_ids_len = input_ids.shape[0]
                    if final_inputs_embeds_len != final_input_ids_len:
                        raise ValueError(
                            f"After insertion, inputs_embeds length "
                            f"({final_inputs_embeds_len}) does not match "
                            f"input_ids length ({final_input_ids_len})"
                        )
        
        # Debug: log shapes before passing to language model
        if not hasattr(self, '_debug_lm_input_logged'):
            print(f"\n[DEBUG llavanext.forward] Before language_model:")
            print(f"  inputs_embeds shape: {inputs_embeds.shape}")
            if attention_mask is not None:
                print(f"  attention_mask shape: {attention_mask.shape}")
            if position_ids is not None:
                print(f"  position_ids shape: {position_ids.shape}")
            if sequence_lengths is not None:
                print(f"  sequence_lengths: {sequence_lengths}")
            self._debug_lm_input_logged = True
        
        # Pass through language model
        # TODO: Implement proper integration with Mistral language model
        # For now, use transformers wrapper
        # Note: transformers language_model expects 2D inputs_embeds (batch, seq_len, hidden_size)
        # but we have varlen format which is 2D [total_tokens, hidden_size]
        # We need to reshape if we have sequence_lengths
        if sequence_lengths is not None and len(sequence_lengths) > 1:
            # Multiple sequences in varlen format - reshape to 2D
            total_tokens = sum(sequence_lengths)
            batch_size = len(sequence_lengths)
            max_seq_len = max(sequence_lengths)
            
            # Check if inputs_embeds is in varlen format
            # It should be 2D [total_tokens, hidden_size]
            if inputs_embeds.shape[0] == total_tokens:
                if inputs_embeds.dim() != 2:
                    raise ValueError(
                        f"inputs_embeds should be 2D [total_tokens, hidden_size], "
                        f"got shape: {inputs_embeds.shape}"
                    )
                hidden_size = inputs_embeds.shape[-1]
                
                # Debug log before reshape
                if not hasattr(self, '_debug_inputs_reshape_logged'):
                    print(f"\n[DEBUG llavanext.forward] Inputs reshape:")
                    print(f"  inputs_embeds (varlen) shape: {inputs_embeds.shape}")
                    print(f"  total_tokens: {total_tokens}")
                    print(f"  batch_size: {batch_size}")
                    print(f"  max_seq_len: {max_seq_len}")
                    print(f"  sequence_lengths: {sequence_lengths}")
                    self._debug_inputs_reshape_logged = True
                
                # Reshape to (batch, max_seq_len, hidden_size) with padding
                inputs_embeds_2d = torch.zeros(
                    batch_size, max_seq_len, hidden_size,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype
                )
                
                # Fill in actual sequences
                cu_seqlens = [0]
                for seq_len in sequence_lengths:
                    cu_seqlens.append(cu_seqlens[-1] + seq_len)
                
                for seq_idx in range(batch_size):
                    seq_start = cu_seqlens[seq_idx]
                    seq_end = cu_seqlens[seq_idx + 1]
                    seq_len = sequence_lengths[seq_idx]
                    # 2D varlen [total_tokens, hidden_size]
                    inputs_embeds_2d[seq_idx, :seq_len] = (
                        inputs_embeds[seq_start:seq_end]
                    )
                
                inputs_embeds = inputs_embeds_2d
                
                # Also reshape attention_mask and position_ids if needed
                if attention_mask is not None:
                    if attention_mask.dim() == 1 and attention_mask.shape[0] == total_tokens:
                        attention_mask_2d = torch.zeros(
                            batch_size, max_seq_len,
                            device=attention_mask.device,
                            dtype=attention_mask.dtype
                        )
                        for seq_idx in range(batch_size):
                            seq_start = cu_seqlens[seq_idx]
                            seq_end = cu_seqlens[seq_idx + 1]
                            seq_len = sequence_lengths[seq_idx]
                            attention_mask_2d[seq_idx, :seq_len] = (
                                attention_mask[seq_start:seq_end]
                            )
                        attention_mask = attention_mask_2d
                
                if position_ids is not None:
                    if position_ids.dim() == 1 and position_ids.shape[0] == total_tokens:
                        position_ids_2d = torch.zeros(
                            batch_size, max_seq_len,
                            device=position_ids.device,
                            dtype=position_ids.dtype
                        )
                        for seq_idx in range(batch_size):
                            seq_start = cu_seqlens[seq_idx]
                            seq_end = cu_seqlens[seq_idx + 1]
                            seq_len = sequence_lengths[seq_idx]
                            position_ids_2d[seq_idx, :seq_len] = (
                                position_ids[seq_start:seq_end]
                            )
                        position_ids = position_ids_2d
        
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        
        # If we reshaped inputs_embeds to batch format, convert hidden_states back to varlen
        if sequence_lengths is not None and len(sequence_lengths) > 1:
            total_tokens = sum(sequence_lengths)
            batch_size = len(sequence_lengths)
            
            # Check if hidden_states is in batch format
            if hidden_states.dim() == 3 and hidden_states.shape[0] == batch_size:
                # hidden_states is batch format (batch, max_seq_len, hidden_size)
                # Convert back to varlen format (total_tokens, hidden_size)
                max_seq_len = hidden_states.shape[1]
                hidden_size = hidden_states.shape[2]
                
                # Calculate cumulative sequence lengths
                cu_seqlens = [0]
                for seq_len in sequence_lengths:
                    cu_seqlens.append(cu_seqlens[-1] + seq_len)
                
                # Create varlen format hidden_states
                hidden_states_varlen = torch.zeros(
                    total_tokens, hidden_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
                
                # Fill in actual sequences (remove padding)
                for seq_idx in range(batch_size):
                    seq_start = cu_seqlens[seq_idx]
                    seq_end = cu_seqlens[seq_idx + 1]
                    seq_len = sequence_lengths[seq_idx]
                    hidden_states_varlen[seq_start:seq_end] = (
                        hidden_states[seq_idx, :seq_len]
                    )
                
                hidden_states = hidden_states_varlen
        
        return hidden_states
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states."""
        return self.lm_head(hidden_states)


def load_llavanext_model(model_path, config):
    """
    Load LLaVANext model
    
    Args:
        model_path: Model path
        config: Configuration object
    
    Returns:
        model: LlavaNextForConditionalGeneration instance
    """
    hf_config = config.hf_config
    
    # Create model
    model = LlavaNextForConditionalGeneration(hf_config)
    
    from nanovllm.utils.loader import load_model

    print("[load_llavanext_model] Loading LLaVANext weights...")

    def name_mapping(weight_name: str) -> str | None:
        # Map transformers weight names to our structure
        if weight_name.startswith("model.vision_tower."):
            # Vision tower weights
            sub_name = weight_name[len("model.vision_tower."):]
            return f"vision_tower.{sub_name}"
        
        if weight_name.startswith("model.multi_modal_projector."):
            # Multimodal projector weights
            sub_name = weight_name[len("model.multi_modal_projector."):]
            return f"multi_modal_projector.{sub_name}"
        
        if weight_name.startswith("model.image_newline"):
            return "image_newline"
        
        if weight_name.startswith("model.language_model."):
            # Language model weights (Mistral)
            sub_name = weight_name[len("model.language_model."):]
            return f"language_model.{sub_name}"
        
        if weight_name.startswith("lm_head."):
            sub_name = weight_name[len("lm_head."):]
            return f"lm_head.{sub_name}"
        
        return None

    load_model(model, model_path, name_mapping=name_mapping)
    return model


__all__ = ["LlavaNextForConditionalGeneration", "load_llavanext_model"]
