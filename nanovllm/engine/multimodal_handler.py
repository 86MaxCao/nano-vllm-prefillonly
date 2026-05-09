"""Handler for multimodal model processing."""
import logging
import torch
from typing import Any

from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class MultimodalHandler:
    """Handles multimodal-specific processing logic."""

    @staticmethod
    def prepare_llavanext_inputs(
        model: torch.nn.Module,
        seqs: list[Sequence],
    ) -> dict[str, Any]:
        """Prepare inputs for llavanext model (no visual method)."""
        pixel_values_list = []
        image_grid_thw_list = []

        for seq in seqs:
            if seq.pixel_values is not None:
                pv = seq.pixel_values
                if isinstance(pv, torch.Tensor) and pv.dim() == 5:
                    b, num_patches, c, h, w = pv.shape
                    if b == 1:
                        pv = pv.squeeze(0)
                    else:
                        pv = pv.view(b * num_patches, c, h, w)
                pixel_values_list.append(pv)
            else:
                pixel_values_list.append(None)

            if seq.image_grid_thw is not None:
                image_grid_thw_list.append(seq.image_grid_thw)
            else:
                image_grid_thw_list.append(None)

        model_kwargs = {}
        model_device = next(model.parameters()).device

        if all(pv is not None for pv in pixel_values_list):
            if len(pixel_values_list) == 1:
                pv = pixel_values_list[0]
                if isinstance(pv, torch.Tensor):
                    pv = pv.to(model_device)
                model_kwargs["pixel_values"] = pv
                if image_grid_thw_list[0] is not None:
                    ig = image_grid_thw_list[0]
                    if isinstance(ig, torch.Tensor):
                        ig = ig.to(model_device)
                    model_kwargs["image_grid_thw"] = ig
            else:
                pv_list_device = []
                for pv in pixel_values_list:
                    if isinstance(pv, torch.Tensor):
                        pv_list_device.append(pv.to(model_device))
                    elif isinstance(pv, list):
                        pv_list_device.append([
                            item.to(model_device) if isinstance(item, torch.Tensor) else item
                            for item in pv
                        ])
                    else:
                        pv_list_device.append(pv)
                model_kwargs["pixel_values"] = pv_list_device
                if all(ig is not None for ig in image_grid_thw_list):
                    ig_list_device = [
                        ig.to(model_device) if isinstance(ig, torch.Tensor) else ig
                        for ig in image_grid_thw_list
                    ]
                    model_kwargs["image_grid_thw"] = ig_list_device

        return model_kwargs

    @staticmethod
    def prepare_prefill_only_inputs(
        model: torch.nn.Module,
        seqs: list[Sequence],
    ) -> dict[str, Any]:
        """Prepare inputs for prefill-only mode (generation, embedding, reranking).

        This method prepares seq_image_indices and seq_vision_placeholders
        which are required for Qwen3-VL models in prefill-only mode.
        """
        pixel_values_list = []
        image_grid_thw_list = []
        seq_image_indices = []
        seq_vision_placeholders = []
        current_image_idx = 0

        for seq_idx, seq in enumerate(seqs):
            seq_start_image_idx = current_image_idx
            if seq.pixel_values is not None:
                pixel_values_list.append(seq.pixel_values)
            else:
                pixel_values_list.append(None)

            if hasattr(seq, 'vision_placeholders') and seq.vision_placeholders:
                seq_vision_placeholders.append(seq.vision_placeholders)
            else:
                seq_vision_placeholders.append([])

            if seq.image_grid_thw is not None:
                image_grid_thw_list.append(seq.image_grid_thw)
                if seq.image_grid_thw.dim() == 1:
                    num_images = 1
                elif seq.image_grid_thw.dim() == 2:
                    num_images = seq.image_grid_thw.shape[0]
                else:
                    num_images = seq.image_grid_thw.shape[0] if seq.image_grid_thw.dim() == 3 else 1
                seq_image_indices.append((seq_start_image_idx, current_image_idx + num_images))
                current_image_idx += num_images
            else:
                image_grid_thw_list.append(None)
                seq_image_indices.append((seq_start_image_idx, current_image_idx))

        model_kwargs = {}
        model_device = next(model.parameters()).device

        # Get vision_dtype directly from the model to avoid double conversion
        vision_dtype = None
        if hasattr(model, 'visual') and hasattr(model.visual, 'vision'):
            try:
                if hasattr(model.visual.vision, 'patch_embed') and hasattr(model.visual.vision.patch_embed, 'proj'):
                    vision_dtype = model.visual.vision.patch_embed.proj.weight.dtype
                elif hasattr(model.visual.vision, 'pos_embed'):
                    vision_dtype = model.visual.vision.pos_embed.weight.dtype
            except AttributeError:
                pass

        if vision_dtype is None:
            vision_dtype = next(model.parameters()).dtype

        if all(pv is not None for pv in pixel_values_list):
            if len(pixel_values_list) == 1:
                pv = pixel_values_list[0]
                if isinstance(pv, torch.Tensor):
                    if pv.device != model_device or pv.dtype != vision_dtype:
                        pv = pv.to(device=model_device, dtype=vision_dtype, non_blocking=True)
                model_kwargs["pixel_values"] = pv
                if image_grid_thw_list[0] is not None:
                    ig = image_grid_thw_list[0]
                    if isinstance(ig, torch.Tensor):
                        if ig.device != model_device:
                            ig = ig.to(model_device, non_blocking=True)
                    model_kwargs["image_grid_thw"] = ig
            else:
                pv_tensors = []
                for idx, pv in enumerate(pixel_values_list):
                    if isinstance(pv, torch.Tensor):
                        if pv.device != model_device or pv.dtype != vision_dtype:
                            pv = pv.to(device=model_device, dtype=vision_dtype, non_blocking=True)
                        pv_tensors.append(pv)
                    else:
                        raise ValueError(f"pixel_values must be torch.Tensor, got {type(pv)}")

                shapes = [pv.shape for pv in pv_tensors]
                if len(set(shapes)) == 1:
                    model_kwargs["pixel_values"] = torch.stack(pv_tensors, dim=0)
                else:
                    model_kwargs["pixel_values"] = torch.cat(pv_tensors, dim=0)

                del pv_tensors

                if all(ig is not None for ig in image_grid_thw_list):
                    ig_tensors = []
                    for ig in image_grid_thw_list:
                        if isinstance(ig, torch.Tensor):
                            if ig.device != model_device:
                                ig = ig.to(model_device, non_blocking=True)
                            ig_tensors.append(ig)
                        else:
                            ig_tensors.append(ig)
                    ig_tensors_2d = []
                    for ig in ig_tensors:
                        if ig.dim() == 1:
                            ig = ig.unsqueeze(0)
                        elif ig.dim() == 3:
                            ig = ig.view(-1, 3)
                        ig_tensors_2d.append(ig)
                    model_kwargs["image_grid_thw"] = torch.cat(ig_tensors_2d, dim=0)

        model_kwargs["vision_slices_per_seq"] = None
        model_kwargs["seq_image_indices"] = seq_image_indices
        model_kwargs["seq_vision_placeholders"] = seq_vision_placeholders

        return model_kwargs
