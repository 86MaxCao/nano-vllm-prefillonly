import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp
import numpy as np

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            use_fast=True,
        )
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._processor = None  # Lazily loaded for multimodal
        atexit.register(self.exit)

    def _get_processor(self):
        """Lazily load and cache the AutoProcessor for multimodal inputs."""
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.model_runner.config.model, trust_remote_code=True
            )
        return self._processor

    def _expand_vision_placeholders(
        self,
        input_ids: list[int],
        image_grid_thw: torch.Tensor,
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        """Expand vision placeholders according to the vision grid metadata."""
        hf_config = self.model_runner.config.hf_config
        vision_config = hf_config.vision_config
        merge_size = vision_config.spatial_merge_size

        image_token_id = getattr(hf_config, "image_token_id", None)
        vision_start_token_id = getattr(
            hf_config,
            "vision_start_token_id",
            None,
        )
        vision_end_token_id = getattr(
            hf_config,
            "vision_end_token_id",
            None,
        )

        if None in (
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
        ):
            raise ValueError(
                "Missing vision placeholder token ids in the config"
            )

        if image_grid_thw.dim() != 2 or image_grid_thw.size(-1) != 3:
            raise ValueError(
                "image_grid_thw must have shape [num_images, 3]"
            )

        grids = image_grid_thw.tolist()
        expected_counts = [
            int(t * h * w // (merge_size**2))
            for t, h, w in grids
        ]

        new_input_ids: list[int] = []
        i = 0
        image_idx = 0
        total_images = len(expected_counts)
        length = len(input_ids)

        placeholder_ranges: list[tuple[int, int]] = []

        while i < length:
            token = input_ids[i]
            if token == vision_start_token_id and image_idx < total_images:
                new_input_ids.append(token)
                i += 1
                # Skip original contents until matching vision_end_token_id
                while i < length and input_ids[i] != vision_end_token_id:
                    i += 1
                if i == length:
                    raise ValueError(
                        "vision_start_token does not have a matching "
                        "vision_end_token"
                    )

                required = expected_counts[image_idx]
                start_offset = len(new_input_ids)
                new_input_ids.extend([image_token_id] * required)
                new_input_ids.append(vision_end_token_id)
                placeholder_ranges.append((start_offset, required))
                i += 1  # Skip the original vision_end token
                image_idx += 1
            else:
                new_input_ids.append(token)
                i += 1

        if image_idx != total_images:
            raise ValueError(
                f"{total_images - image_idx} images do not have matching "
                "placeholders"
            )

        return new_input_ids, expected_counts, placeholder_ranges

    def exit(self):
        if hasattr(self, "model_runner") and self.model_runner is not None:
            self.model_runner.call("exit")
            del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        images=None,
        pixel_values=None,
        image_grid_thw=None,
        vision_counts=None,
        vision_placeholders=None,
    ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(
            prompt,
            sampling_params,
            images=images,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vision_counts=vision_counts,
            vision_placeholders=vision_placeholders,
        )
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        # Clean up GDN states for finished sequences
        finished_seq_ids = [seq.seq_id for seq in seqs if seq.is_finished]
        if finished_seq_ids:
            self.model_runner.call("cleanup_seq_states", finished_seq_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        num_tokens = (
            sum(len(seq) for seq in seqs)
            if is_prefill
            else -len(seqs)
        )
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(
                total=len(prompts),
                desc="Generating",
                dynamic_ncols=True,
            )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = (
                        num_tokens / (perf_counter() - t)
                    )
                else:
                    decode_throughput = (
                        -num_tokens / (perf_counter() - t)
                    )
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [
            outputs[seq_id]
            for seq_id in sorted(outputs.keys())
        ]
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
            }
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs

    def generate_multimodal(
        self,
        requests: list[dict],
        sampling_params: SamplingParams | list[SamplingParams],
        processor,
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(
                total=len(requests),
                desc="Generating",
                dynamic_ncols=True,
            )

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(requests)

        # Phase 1: Extract text and images from all requests
        all_texts = []
        all_images = []  # per-request image list (None if no images)
        for request in requests:
            messages = request.get("messages")
            text = request.get("text")
            images = request.get("images")

            if text is None:
                if messages is None:
                    raise ValueError(
                        "multimodal request requires 'text' or 'messages'"
                    )
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if images is None:
                    extracted_images = []
                    for message in messages:
                        for content in message.get("content", []):
                            is_image = content.get("type") == "image"
                            has_payload = "image" in content
                            if is_image and has_payload:
                                extracted_images.append(content["image"])
                    images = extracted_images if extracted_images else None

            if images is not None and not isinstance(images, (list, tuple)):
                images = [images]

            all_texts.append(text)
            all_images.append(images)

        # Phase 2: Batch processor call
        # Separate multimodal vs text-only requests
        multimodal_indices = [i for i, imgs in enumerate(all_images) if imgs]
        text_only_indices = [i for i, imgs in enumerate(all_images) if not imgs]

        # Per-request results storage
        per_request_input_ids = [None] * len(requests)
        per_request_pixel_values = [None] * len(requests)
        per_request_image_grid_thw = [None] * len(requests)

        if multimodal_indices:
            mm_texts = [all_texts[i] for i in multimodal_indices]
            # Flatten images for processor: one image per entry for single-image requests
            # Track how many images each request contributes
            mm_flat_images = []
            images_per_request = []
            for i in multimodal_indices:
                imgs = all_images[i]
                mm_flat_images.extend(imgs)
                images_per_request.append(len(imgs))

            processor_outputs = processor(
                text=mm_texts,
                images=mm_flat_images,
                return_tensors="pt",
                padding=True,
            )

            batch_input_ids = processor_outputs["input_ids"]
            attention_mask = processor_outputs.get("attention_mask")
            pixel_values = processor_outputs.get("pixel_values")
            image_grid_thw = processor_outputs.get("image_grid_thw")

            # Handle LlavaNext 1D pixel_values bug
            if pixel_values is not None and isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 1:
                import logging
                logging.warning(f"LlavaNextProcessor returned 1D pixel_values {pixel_values.shape}, re-processing")
                processor_outputs_fix = processor(
                    text=mm_texts,
                    images=mm_flat_images,
                    padding=True,
                )
                pixel_values = processor_outputs_fix.get("pixel_values")
                if isinstance(pixel_values, list):
                    pixel_values = [
                        torch.tensor(pv) if not isinstance(pv, torch.Tensor) else pv
                        for pv in pixel_values
                    ]
                else:
                    raise ValueError(
                        f"Failed to fix pixel_values format: {type(pixel_values)}"
                    )
                image_grid_thw = processor_outputs_fix.get("image_grid_thw")
                batch_input_ids = processor_outputs_fix["input_ids"]
                if isinstance(batch_input_ids, list):
                    batch_input_ids = torch.tensor(batch_input_ids)
                attention_mask = processor_outputs_fix.get("attention_mask")
                if isinstance(attention_mask, list):
                    attention_mask = torch.tensor(attention_mask)

            # Convert image_grid_thw to tensor if needed
            if image_grid_thw is not None and not isinstance(image_grid_thw, torch.Tensor):
                if isinstance(image_grid_thw, list):
                    image_grid_thw = torch.tensor(image_grid_thw)
                elif isinstance(image_grid_thw, np.ndarray):
                    image_grid_thw = torch.from_numpy(image_grid_thw)

            # Compute patches-per-image for pixel_values splitting
            # For Qwen3-VL: pixel_values is [total_patches, hidden_dim]
            # where patches per image = t * h * w from image_grid_thw
            patches_per_image = None
            if image_grid_thw is not None and isinstance(pixel_values, torch.Tensor):
                thw_2d = image_grid_thw.squeeze(0) if image_grid_thw.dim() == 3 else image_grid_thw
                if thw_2d.dim() == 2:
                    patches_per_image = (thw_2d[:, 0] * thw_2d[:, 1] * thw_2d[:, 2]).tolist()

            # Unbatch: distribute results back to per-request storage
            img_offset = 0
            patch_offset = 0
            for batch_idx, req_idx in enumerate(multimodal_indices):
                # Extract unpadded input_ids using attention_mask
                if attention_mask is not None:
                    mask = attention_mask[batch_idx].bool()
                    ids = batch_input_ids[batch_idx][mask].tolist()
                else:
                    ids = batch_input_ids[batch_idx].tolist()
                per_request_input_ids[req_idx] = ids

                # Slice pixel_values for this request
                num_imgs = images_per_request[batch_idx]
                if pixel_values is not None:
                    if isinstance(pixel_values, list):
                        # LlavaNext: list of tensors per image
                        per_request_pixel_values[req_idx] = [
                            pv.cpu() if isinstance(pv, torch.Tensor) else pv
                            for pv in pixel_values[img_offset:img_offset + num_imgs]
                        ]
                    elif isinstance(pixel_values, torch.Tensor):
                        if patches_per_image is not None:
                            # Qwen-VL style: pixel_values is [total_patches, hidden_dim]
                            # Split by patch count per image
                            total_patches = sum(patches_per_image[img_offset:img_offset + num_imgs])
                            pv_slice = pixel_values[patch_offset:patch_offset + total_patches]
                            per_request_pixel_values[req_idx] = pv_slice.contiguous().cpu()
                            patch_offset += total_patches
                        else:
                            # Fallback: assume dim 0 is image count
                            pv_slice = pixel_values[img_offset:img_offset + num_imgs]
                            per_request_pixel_values[req_idx] = pv_slice.contiguous().cpu()

                # Slice image_grid_thw for this request
                if image_grid_thw is not None:
                    thw = image_grid_thw
                    if thw.dim() == 3:
                        thw = thw.squeeze(0)
                    if thw.dim() == 2:
                        per_request_image_grid_thw[req_idx] = thw[img_offset:img_offset + num_imgs].contiguous().cpu()
                    else:
                        per_request_image_grid_thw[req_idx] = thw.contiguous().cpu()

                img_offset += num_imgs

        if text_only_indices:
            to_texts = [all_texts[i] for i in text_only_indices]
            text_outputs = self.tokenizer(
                to_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            text_input_ids = text_outputs["input_ids"]
            text_attention_mask = text_outputs.get("attention_mask")
            for batch_idx, req_idx in enumerate(text_only_indices):
                if text_attention_mask is not None:
                    mask = text_attention_mask[batch_idx].bool()
                    ids = text_input_ids[batch_idx][mask].tolist()
                else:
                    ids = text_input_ids[batch_idx].tolist()
                per_request_input_ids[req_idx] = ids

        # Phase 3: Expand vision placeholders and add_request for each sequence
        for req_idx, sp in enumerate(sampling_params):
            input_ids = per_request_input_ids[req_idx]
            pixel_values = per_request_pixel_values[req_idx]
            image_grid_thw = per_request_image_grid_thw[req_idx]

            vision_counts = []
            vision_placeholders = []
            if image_grid_thw is not None:
                (
                    expanded_input_ids,
                    vision_counts,
                    vision_placeholders,
                ) = self._expand_vision_placeholders(
                    input_ids,
                    image_grid_thw.squeeze(0)
                    if image_grid_thw.dim() == 3
                    else image_grid_thw,
                )
                input_ids = expanded_input_ids

            self.add_request(
                input_ids,
                sp,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                vision_counts=vision_counts,
                vision_placeholders=vision_placeholders,
            )

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        outputs = [
            outputs[seq_id]
            for seq_id in sorted(outputs.keys())
        ]
        results = [
            {
                # Decode without special tokens so the response is clean.
                "text": self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "token_ids": token_ids,
            }
            for token_ids in outputs
        ]

        if use_tqdm:
            pbar.close()

        return results

    def embed_batch(
        self,
        texts: list[str] | list[list[int]],
        images: list | None = None,
        use_tqdm: bool = False,
    ) -> torch.Tensor:
        """Batch embedding generation (prefill-only).

        Args:
            texts: List of text prompts or token IDs
            images: Optional list of images for multimodal embedding
            use_tqdm: Whether to show progress bar

        Returns:
            embeddings: Tensor of shape [batch_size, hidden_size]
        """
        if not self.model_runner.is_embedding:
            msg = "Model is not configured for embedding. Set is_embedding=True."
            raise ValueError(msg)

        if use_tqdm:
            pbar = tqdm(total=len(texts), desc="Embedding", dynamic_ncols=True)

        # Set tokenizer padding side to left for consistent last token extraction
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        batch_size = len(texts)

        # Handle multimodal case
        if images is not None:
            # Multimodal embedding: use cached processor
            processor = self._get_processor()
            # Set padding side to left
            if hasattr(processor, 'tokenizer'):
                processor.tokenizer.padding_side = "left"

            # Collect all formatted texts and images using batch apply_chat_template
            all_images_list = []
            messages_batch = []
            text_only_fallback_indices = []
            
            for i, text in enumerate(texts):
                if images and i < len(images):
                    img = images[i] if isinstance(images[i], list) else [images[i]]
                    messages_batch.append([
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img[0]},
                                {"type": "text", "text": text},
                            ],
                        }
                    ])
                    all_images_list.append(img[0])
                else:
                    messages_batch.append([
                        {"role": "user", "content": [{"type": "text", "text": text}]}
                    ])
                    all_images_list.append(None)
                    text_only_fallback_indices.append(i)
            
            # Batch apply_chat_template (much faster than serial calls)
            try:
                all_formatted_texts = processor.apply_chat_template(
                    messages_batch, tokenize=False, add_generation_prompt=False,
                )
            except (AttributeError, TypeError):
                # Fallback to serial if batch not supported
                all_formatted_texts = []
                for msgs in messages_batch:
                    all_formatted_texts.append(
                        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    )
            
            # Qwen3 embedding models use <|endoftext|> as pooling anchor
            embed_type = getattr(self.model_runner.config, "embedding_type", None)
            if embed_type in ("qwen3", "qwen3_vl"):
                all_formatted_texts = [t + "<|endoftext|>" for t in all_formatted_texts]
            
            # Filter out None images and process in batches
            # Separate multimodal and text-only samples
            multimodal_texts = []
            multimodal_images = []
            text_only_texts = []
            multimodal_indices = []
            text_only_indices = []
            
            for i, (text, img) in enumerate(zip(all_formatted_texts, all_images_list)):
                if img is not None:
                    multimodal_texts.append(text)
                    multimodal_images.append(img)
                    multimodal_indices.append(i)
                else:
                    text_only_texts.append(text)
                    text_only_indices.append(i)
            
            # Process multimodal samples in batch
            if multimodal_texts:
                processor_outputs = processor(
                    text=multimodal_texts,
                    images=multimodal_images,
                    return_tensors="pt",
                    padding=True,
                )
                multimodal_input_ids = processor_outputs["input_ids"]
                multimodal_attention_mask = processor_outputs.get("attention_mask")
                multimodal_pixel_values = processor_outputs.get("pixel_values")
                multimodal_image_grid_thw = processor_outputs.get("image_grid_thw")
                
                # Handle image_grid_thw shape
                if multimodal_image_grid_thw is not None:
                    if multimodal_image_grid_thw.dim() == 3 and multimodal_image_grid_thw.shape[1] == 1:
                        multimodal_image_grid_thw = multimodal_image_grid_thw.squeeze(1)
            else:
                multimodal_input_ids = None
                multimodal_attention_mask = None
                multimodal_pixel_values = None
                multimodal_image_grid_thw = None
            
            # Process text-only samples in batch
            if text_only_texts:
                text_tokenized = self.tokenizer(
                    text_only_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                text_only_input_ids = text_tokenized["input_ids"]
                text_only_attention_mask = text_tokenized.get("attention_mask")
            else:
                text_only_input_ids = None
                text_only_attention_mask = None
            
            # Merge multimodal and text-only results
            if multimodal_input_ids is not None and text_only_input_ids is not None:
                # Need to merge: create full batch tensors
                max_len = max(
                    multimodal_input_ids.shape[1],
                    text_only_input_ids.shape[1]
                )
                # Pad both to max_len
                if multimodal_input_ids.shape[1] < max_len:
                    pad_len = max_len - multimodal_input_ids.shape[1]
                    pad_token = self.tokenizer.pad_token_id or 0
                    pad_tensor = torch.full(
                        (multimodal_input_ids.shape[0], pad_len),
                        pad_token,
                        dtype=multimodal_input_ids.dtype
                    )
                    multimodal_input_ids = torch.cat([pad_tensor, multimodal_input_ids], dim=1)
                    if multimodal_attention_mask is not None:
                        pad_mask = torch.zeros(
                            (multimodal_attention_mask.shape[0], pad_len),
                            dtype=multimodal_attention_mask.dtype
                        )
                        multimodal_attention_mask = torch.cat([pad_mask, multimodal_attention_mask], dim=1)
                
                if text_only_input_ids.shape[1] < max_len:
                    pad_len = max_len - text_only_input_ids.shape[1]
                    pad_token = self.tokenizer.pad_token_id or 0
                    pad_tensor = torch.full(
                        (text_only_input_ids.shape[0], pad_len),
                        pad_token,
                        dtype=text_only_input_ids.dtype
                    )
                    text_only_input_ids = torch.cat([pad_tensor, text_only_input_ids], dim=1)
                    if text_only_attention_mask is not None:
                        pad_mask = torch.zeros(
                            (text_only_attention_mask.shape[0], pad_len),
                            dtype=text_only_attention_mask.dtype
                        )
                        text_only_attention_mask = torch.cat([pad_mask, text_only_attention_mask], dim=1)
                
                # Merge in original order
                input_ids_list = [None] * batch_size
                attention_mask_list = [None] * batch_size
                pixel_values_list = [None] * batch_size
                image_grid_thw_list = [None] * batch_size
                
                for idx, orig_idx in enumerate(multimodal_indices):
                    input_ids_list[orig_idx] = multimodal_input_ids[idx]
                    if multimodal_attention_mask is not None:
                        attention_mask_list[orig_idx] = multimodal_attention_mask[idx]
                    if multimodal_pixel_values is not None:
                        pixel_values_list[orig_idx] = multimodal_pixel_values[idx]
                    if multimodal_image_grid_thw is not None:
                        image_grid_thw_list[orig_idx] = multimodal_image_grid_thw[idx]
                
                for idx, orig_idx in enumerate(text_only_indices):
                    input_ids_list[orig_idx] = text_only_input_ids[idx]
                    if text_only_attention_mask is not None:
                        attention_mask_list[orig_idx] = text_only_attention_mask[idx]
                
                input_ids_tensor = torch.stack(input_ids_list)
                if attention_mask_list[0] is not None:
                    attention_mask_tensor = torch.stack(attention_mask_list)
                else:
                    attention_mask_tensor = None
                
                # Handle pixel_values and image_grid_thw
                if any(pv is not None for pv in pixel_values_list):
                    # Ensure each pv is 2D (num_patches, cps) before concatenating
                    pv_list = []
                    for pv in pixel_values_list:
                        if pv is not None:
                            # If 3D, squeeze or reshape to 2D
                            if pv.dim() == 3:
                                # (1, num_patches, cps) -> (num_patches, cps)
                                if pv.shape[0] == 1:
                                    pv = pv.squeeze(0)
                                else:
                                    # (batch, num_patches, cps) -> (batch*num_patches, cps)
                                    pv = pv.view(-1, pv.shape[-1])
                            elif pv.dim() > 2:
                                # Other high-dim cases, reshape to 2D
                                pv = pv.view(-1, pv.shape[-1])
                            pv_list.append(pv)
                    pixel_values_batch = torch.cat(pv_list, dim=0) if pv_list else None
                else:
                    pixel_values_batch = None
                
                if any(thw is not None for thw in image_grid_thw_list):
                    image_grid_thw_batch = torch.stack([thw for thw in image_grid_thw_list if thw is not None], dim=0)
                else:
                    image_grid_thw_batch = None
            elif multimodal_input_ids is not None:
                input_ids_tensor = multimodal_input_ids
                attention_mask_tensor = multimodal_attention_mask
                # Ensure pixel_values is 2D (total_patches, cps) format
                if multimodal_pixel_values is not None:
                    if multimodal_pixel_values.dim() == 3:
                        # (batch, num_patches, cps) -> (batch*num_patches, cps)
                        # But wait, this might be wrong - processor might return per-image format
                        # Check if it's already flattened or needs flattening
                        if multimodal_pixel_values.shape[0] == len(multimodal_texts):
                            # Likely (batch, num_patches_per_image, cps) - need to flatten
                            pixel_values_batch = multimodal_pixel_values.view(-1, multimodal_pixel_values.shape[-1])
                        else:
                            # Already in correct format or single image
                            pixel_values_batch = multimodal_pixel_values
                    elif multimodal_pixel_values.dim() == 2:
                        # Already in correct format (total_patches, cps)
                        pixel_values_batch = multimodal_pixel_values
                    else:
                        # Other cases, try to reshape to 2D
                        pixel_values_batch = multimodal_pixel_values.view(-1, multimodal_pixel_values.shape[-1])
                else:
                    pixel_values_batch = None
                image_grid_thw_batch = multimodal_image_grid_thw
            else:
                input_ids_tensor = text_only_input_ids
                attention_mask_tensor = text_only_attention_mask
                pixel_values_batch = None
                image_grid_thw_batch = None
            
            # Convert to int64 if needed
            if input_ids_tensor.dtype != torch.int64:
                input_ids_tensor = input_ids_tensor.to(torch.int64)
            if attention_mask_tensor is not None and attention_mask_tensor.dtype != torch.int64:
                attention_mask_tensor = attention_mask_tensor.to(torch.int64)
        else:
            # Text-only embedding: use tokenizer batch processing
            # Convert token IDs to strings if needed
            all_texts = []
            for text in texts:
                if isinstance(text, str):
                    all_texts.append(text)
                else:
                    # Decode token IDs to string
                    all_texts.append(self.tokenizer.decode(text))
            
            # Qwen3 embedding models use <|endoftext|> as pooling anchor
            embed_type = getattr(self.model_runner.config, "embedding_type", None)
            if embed_type in ("qwen3", "qwen3_vl"):
                all_texts = [t + "<|endoftext|>" for t in all_texts]
            
            # Batch tokenize with left padding
            tokenized = self.tokenizer(
                all_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            
            input_ids_tensor = tokenized["input_ids"].to(torch.int64)
            attention_mask_tensor = tokenized.get("attention_mask")
            if attention_mask_tensor is not None:
                attention_mask_tensor = attention_mask_tensor.to(torch.int64)
            pixel_values_batch = None
            image_grid_thw_batch = None
        
        # Generate positions tensor
        max_len = input_ids_tensor.shape[1]
        positions_tensor = torch.arange(max_len, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1)
        
        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        # Call embed method
        embeddings = self.model_runner.call(
            "embed",
            input_ids_tensor,
            positions_tensor,
            attention_mask_tensor,
            pixel_values=pixel_values_batch,
            image_grid_thw=image_grid_thw_batch,
        )

        if use_tqdm:
            pbar.close()

        return embeddings

    def rerank_batch(
        self,
        query_doc_pairs: list[tuple[str, str]] | list[tuple[list[int], list[int]]],
        images: list | None = None,
        use_tqdm: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch reranking (prefill-only).

        Args:
            query_doc_pairs: List of (query, document) pairs
            images: Optional list of images for multimodal reranking
            use_tqdm: Whether to show progress bar

        Returns:
            For pointwise rerankers: scores tensor [batch_size]
            For listwise rerankers: (scores, query_embeds, doc_embeds)
        """
        if not self.model_runner.is_reranker:
            msg = "Model is not configured for reranking. Set is_reranker=True."
            raise ValueError(msg)

        if use_tqdm:
            pbar = tqdm(
                total=len(query_doc_pairs), desc="Reranking", dynamic_ncols=True
            )

        # Set tokenizer padding side to left for consistent last token extraction
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        batch_size = len(query_doc_pairs)

        # Check if we have a processor for multimodal reranking
        processor = None
        if images:
            try:
                processor = self._get_processor()
                # Set padding side to left to match Transformers baseline
                if hasattr(processor, 'tokenizer'):
                    processor.tokenizer.padding_side = "left"
            except Exception:
                processor = None

        # Batch processing: collect all inputs first
        if images and processor:
            # Multimodal reranking: use batch apply_chat_template
            all_images_list = []
            messages_batch = []
            
            for i, (query, doc) in enumerate(query_doc_pairs):
                messages_batch.append([
                    {
                        "role": "system",
                        "content": 'Judge whether the Document is relevant to the Query. Answer only "yes" or "no".',
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": images[i]},
                            {"type": "text", "text": f"<query>{query}</query>\n<document>{doc}</document>"},
                        ],
                    }
                ])
                all_images_list.append(images[i])
            
            # Batch apply_chat_template (much faster than serial calls)
            try:
                all_formatted_texts = processor.apply_chat_template(
                    messages_batch, tokenize=False, add_generation_prompt=True,
                )
            except (AttributeError, TypeError):
                # Fallback to serial if batch not supported
                all_formatted_texts = []
                for msgs in messages_batch:
                    all_formatted_texts.append(
                        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    )
            
            # Batch process all multimodal inputs at once
            processor_outputs = processor(
                text=all_formatted_texts,
                images=all_images_list,
                return_tensors="pt",
                padding=True,  # Processor will handle batch padding
            )
            
            input_ids_tensor = processor_outputs["input_ids"]
            attention_mask_tensor = processor_outputs.get("attention_mask")
            pixel_values_batch = processor_outputs.get("pixel_values")
            image_grid_thw_batch = processor_outputs.get("image_grid_thw")
            
            # Handle image_grid_thw shape: squeeze [1, 3] to [3] if needed
            if image_grid_thw_batch is not None:
                if image_grid_thw_batch.dim() == 3 and image_grid_thw_batch.shape[1] == 1:
                    image_grid_thw_batch = image_grid_thw_batch.squeeze(1)  # [batch, 1, 3] -> [batch, 3]
            
            # Convert to int64 if needed
            if input_ids_tensor.dtype != torch.int64:
                input_ids_tensor = input_ids_tensor.to(torch.int64)
            if attention_mask_tensor is not None and attention_mask_tensor.dtype != torch.int64:
                attention_mask_tensor = attention_mask_tensor.to(torch.int64)
        else:
            # Text-only reranking: use tokenizer batch processing
            reranker_type = getattr(self.model_runner.config, "reranker_type", None)
            all_texts = []
            if reranker_type in ("qwen3", "qwen3_vl"):
                # Batch apply_chat_template for Qwen3 rerankers
                messages_batch = []
                for query, doc in query_doc_pairs:
                    if isinstance(query, str) and isinstance(doc, str):
                        messages_batch.append([
                            {"role": "system", "content": 'Judge whether the Document is relevant to the Query. Answer only "yes" or "no".'},
                            {"role": "user", "content": f"<query>{query}</query>\n<document>{doc}</document>"},
                        ])
                    else:
                        # Decode token IDs first
                        q = self.tokenizer.decode(query) if isinstance(query, list) else query
                        d = self.tokenizer.decode(doc) if isinstance(doc, list) else doc
                        messages_batch.append([
                            {"role": "system", "content": 'Judge whether the Document is relevant to the Query. Answer only "yes" or "no".'},
                            {"role": "user", "content": f"<query>{q}</query>\n<document>{d}</document>"},
                        ])
                try:
                    all_texts = self.tokenizer.apply_chat_template(
                        messages_batch, tokenize=False, add_generation_prompt=True
                    )
                except (AttributeError, TypeError):
                    all_texts = [
                        self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        for msgs in messages_batch
                    ]
            else:
                for query, doc in query_doc_pairs:
                    if isinstance(query, str) and isinstance(doc, str):
                        # Other rerankers (jina, gemma, etc.) use simple format
                        sep_token = (
                            self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
                        )
                        text = f"{query}{self.tokenizer.decode([sep_token])}{doc}"
                        all_texts.append(text)
                    else:
                        # If already token IDs, decode them
                        if isinstance(query, list) and isinstance(doc, list):
                            sep_token = (
                                self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
                            )
                            combined_ids = query + [sep_token] + doc
                            text = self.tokenizer.decode(combined_ids)
                            all_texts.append(text)
                        else:
                            raise ValueError(
                                "Mixed string and token ID inputs not supported"
                            )
            
            # Batch tokenize with left padding
            tokenized = self.tokenizer(
                all_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            
            input_ids_tensor = tokenized["input_ids"].to(torch.int64)
            attention_mask_tensor = tokenized.get("attention_mask")
            if attention_mask_tensor is not None:
                attention_mask_tensor = attention_mask_tensor.to(torch.int64)
            pixel_values_batch = None
            image_grid_thw_batch = None
        
        # Generate positions tensor
        max_len = input_ids_tensor.shape[1]
        positions_tensor = torch.arange(max_len, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1)
        
        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        # Call rerank method
        result = self.model_runner.call(
            "rerank",
            input_ids_tensor,
            positions_tensor,
            None,  # token_indices
            attention_mask_tensor,
            False,  # use_flex_attention
            pixel_values_batch,  # pixel_values
            image_grid_thw_batch,  # image_grid_thw
        )

        if use_tqdm:
            pbar.close()

        return result

    def generate_single_token(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | None = None,
        images: list | None = None,
        use_tqdm: bool = False,
    ) -> list[int]:
        """Single token generation (prefill + one sample, optimized path).

        Args:
            prompts: List of text prompts or token IDs
            sampling_params: Sampling parameters (max_tokens set to 1)
            images: Optional list of images for multimodal generation
            use_tqdm: Whether to show progress bar

        Returns:
            List of generated token IDs (one per prompt)
        """
        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=1)
        else:
            # Force max_tokens to 1 for single token generation
            sampling_params.max_tokens = 1

        if use_tqdm:
            pbar = tqdm(
                total=len(prompts), desc="Single Token", dynamic_ncols=True
            )

        if not isinstance(sampling_params, list):
            sampling_params_list = [sampling_params] * len(prompts)
        else:
            sampling_params_list = sampling_params

        # Add all requests
        for i, prompt in enumerate(prompts):
            sp = sampling_params_list[i]
            sp.max_tokens = 1  # Ensure single token
            if images and i < len(images):
                # For multimodal, use generate_multimodal
                # This is a simplified version - full multimodal support
                # would need proper request formatting
                img = images[i] if isinstance(images[i], list) else [images[i]]
                self.add_request(prompt, sp, images=img)
            else:
                self.add_request(prompt, sp)

        outputs = {}
        while not self.is_finished():
            output, _ = self.step()
            for seq_id, token_ids in output:
                # Get first (and only) token
                outputs[seq_id] = token_ids[0] if token_ids else None
                if use_tqdm:
                    pbar.update(1)

        if use_tqdm:
            pbar.close()

        # Return in order
        result = [
            outputs.get(seq_id, None) for seq_id in sorted(outputs.keys())
        ]
        return result