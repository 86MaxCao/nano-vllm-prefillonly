"""Processor for multimodal generation tasks."""
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class MultimodalProcessor:
    """Handles multimodal input processing for generation."""

    @staticmethod
    def process_multimodal_request(
        request: dict,
        processor: Any,
        tokenizer: Any,
    ) -> tuple[list[int], torch.Tensor | None, torch.Tensor | None]:
        """Process a single multimodal request.

        Returns:
            tuple: (input_ids, pixel_values, image_grid_thw)
        """
        messages = request.get("messages")
        text = request.get("text")
        images = request.get("images")

        if text is None:
            if messages is None:
                raise ValueError("multimodal request requires 'text' or 'messages'")
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if images is None:
                images = MultimodalProcessor._extract_images_from_messages(messages)

        if images is not None and not isinstance(images, (list, tuple)):
            images = [images]

        processor_kwargs = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            processor_kwargs["images"] = images

        processor_outputs = processor(**processor_kwargs)
        input_ids = processor_outputs["input_ids"][0].tolist()
        pixel_values = processor_outputs.get("pixel_values")
        image_grid_thw = processor_outputs.get("image_grid_thw")

        # Fix: LlavaNextProcessor with return_tensors="pt" sometimes returns 1D tensor
        if pixel_values is not None and isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 1:
            logger.warning(
                "LlavaNextProcessor returned 1D pixel_values tensor with shape %s. "
                "Attempting to fix by re-processing without return_tensors='pt'",
                pixel_values.shape
            )
            pixel_values, image_grid_thw = MultimodalProcessor._fix_pixel_values(
                processor, text, images
            )

        # Convert image_grid_thw to tensor if needed
        if image_grid_thw is not None and not isinstance(image_grid_thw, torch.Tensor):
            if isinstance(image_grid_thw, list):
                image_grid_thw = torch.tensor(image_grid_thw)
            elif hasattr(image_grid_thw, '__array__'):
                import numpy as np
                if isinstance(image_grid_thw, np.ndarray):
                    image_grid_thw = torch.from_numpy(image_grid_thw)

        # Handle pixel_values format
        if pixel_values is not None:
            pixel_values = MultimodalProcessor._normalize_pixel_values(pixel_values)
            pixel_values = pixel_values.contiguous().cpu()

        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.contiguous().cpu()

        return input_ids, pixel_values, image_grid_thw

    @staticmethod
    def _extract_images_from_messages(messages: list[dict]) -> list | None:
        """Extract images from messages."""
        extracted_images = []
        for message in messages:
            for content in message.get("content", []):
                is_image = content.get("type") == "image"
                has_payload = "image" in content
                if is_image and has_payload:
                    extracted_images.append(content["image"])
        return extracted_images if extracted_images else None

    @staticmethod
    def _fix_pixel_values(
        processor: Any,
        text: str,
        images: list | None,
    ) -> tuple[torch.Tensor | list | None, torch.Tensor | None]:
        """Fix pixel_values format by re-processing."""
        processor_kwargs_fix = {
            "text": [text],
            "padding": True,
        }
        if images:
            processor_kwargs_fix["images"] = images
        processor_outputs_fix = processor(**processor_kwargs_fix)
        pixel_values_fix = processor_outputs_fix.get("pixel_values")

        if isinstance(pixel_values_fix, list):
            pixel_values = [
                torch.tensor(pv) if not isinstance(pv, torch.Tensor) else pv
                for pv in pixel_values_fix
            ]
            logger.info("Successfully fixed pixel_values to list format with %d elements", len(pixel_values))
            return pixel_values, processor_outputs_fix.get("image_grid_thw")
        else:
            raise ValueError(
                f"Failed to fix pixel_values format. "
                f"Original: 1D tensor. Fixed: {type(pixel_values_fix)}"
            )

    @staticmethod
    def _normalize_pixel_values(pixel_values: Any) -> torch.Tensor | list:
        """Normalize pixel_values to correct format."""
        if isinstance(pixel_values, list):
            return pixel_values
        elif isinstance(pixel_values, torch.Tensor):
            if pixel_values.dim() == 1:
                raise ValueError(
                    f"Received 1D pixel_values tensor with shape {pixel_values.shape}. "
                    f"This usually happens when processor output is incorrectly converted."
                )
            return pixel_values
        else:
            logger.warning("pixel_values has unexpected type: %s", type(pixel_values))
            if hasattr(pixel_values, 'cpu'):
                return pixel_values.cpu()
            return pixel_values
