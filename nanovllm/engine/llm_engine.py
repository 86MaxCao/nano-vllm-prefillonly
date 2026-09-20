import atexit
import itertools
import os
import socket
from contextlib import contextmanager
from dataclasses import fields, replace
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


def _allocate_rendezvous() -> tuple[str, int]:
    """Pick the MASTER_ADDR/MASTER_PORT all ranks will rendezvous on.

    Every rank must join the *same* process-group store; a per-rank random
    port used to create N unrelated stores and deadlock initialization.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1", sock.getsockname()[1]
    finally:
        sock.close()


def _check_batch_lengths(lengths: dict[str, int]) -> None:
    """Reject mismatched per-request lists instead of silently truncating.

    `zip(prompts, sampling_params)` used to drop requests when one list was
    shorter, returning a short batch that looked successful.
    """
    unique = set(lengths.values())
    if len(unique) > 1:
        details = ", ".join(f"{name}={n}" for name, n in lengths.items())
        raise ValueError(
            f"Batch inputs have mismatched lengths ({details}); requests "
            "would be silently dropped."
        )


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        unknown_kwargs = {k for k in kwargs if k not in config_fields}
        if unknown_kwargs:
            raise TypeError(
                f"Unknown LLMEngine arguments: {sorted(unknown_kwargs)}. "
                f"Valid arguments are: {sorted(config_fields)}."
            )
        config = Config(model, **kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        if config.tensor_parallel_size > 1:
            # One rendezvous address shared by every rank; per-rank random
            # ports used to create unrelated stores and deadlock init.
            master_addr, master_port = _allocate_rendezvous()
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(master_port)
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        try:
            self.model_runner = ModelRunner(config, 0, self.events)
        except Exception:
            # Rank 0 failed after workers were spawned; without this cleanup
            # the workers would block forever on their rendezvous/barrier.
            for p in self.ps:
                if p.is_alive():
                    p.terminate()
                p.join()
            raise
        # config.model is always a local directory by this point (Config
        # resolves Hub ids via snapshot_download with the pinned revision),
        # so no revision kwarg is needed here.
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            use_fast=True,
        )
        # Prefer the config's stop ids, which may list several; fall back to the
        # tokenizer only when the config did not provide any.
        if config.eos in (-1, None) and self.tokenizer.eos_token_id is not None:
            config.eos = self.tokenizer.eos_token_id
        self.config = config
        self.scheduler = Scheduler(config)
        self._processor = None  # Lazily loaded for multimodal
        atexit.register(self.exit)

    @contextmanager
    def _left_padding(self):
        """Temporarily left-pad, restoring the previous setting on any exit.

        Last-token pooling relies on left padding; restoring in a finally block
        keeps a mid-batch failure from corrupting later tokenization.
        """
        original = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        processor_tokenizer = getattr(self._processor, "tokenizer", None)
        processor_original = (
            processor_tokenizer.padding_side if processor_tokenizer else None
        )
        if processor_tokenizer is not None:
            processor_tokenizer.padding_side = "left"
        try:
            yield
        finally:
            self.tokenizer.padding_side = original
            if processor_tokenizer is not None:
                processor_tokenizer.padding_side = processor_original

    def _get_processor(self):
        """Lazily load and cache the AutoProcessor for multimodal inputs."""
        if self._processor is None:
            from transformers import AutoProcessor
            # config.model is always a local directory (Config resolves Hub
            # ids via snapshot_download with the pinned revision), so no
            # revision kwarg is needed here.
            processor_kwargs = {
                "trust_remote_code": self.model_runner.config.trust_remote_code,
            }
            self._processor = AutoProcessor.from_pretrained(
                self.model_runner.config.model,
                **processor_kwargs,
            )
        return self._processor

    @staticmethod
    def _normalize_pixel_values(pixel_values, num_texts: int):
        """Normalise processor pixel_values to 2-D (total_patches, channels).

        Some processors return (num_images, num_patches, channels); the model
        runner expects the patches of all images concatenated along dim 0.
        """
        if pixel_values is None:
            return None
        if pixel_values.dim() == 3 and pixel_values.shape[0] == num_texts:
            return pixel_values.reshape(-1, pixel_values.shape[-1])
        if pixel_values.dim() == 2:
            return pixel_values
        return pixel_values.reshape(-1, pixel_values.shape[-1])

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
        if not prompt:
            raise ValueError("Prompt is empty after tokenization.")
        if len(prompt) > self.scheduler.max_num_batched_tokens:
            raise ValueError(
                f"Prompt of {len(prompt)} tokens exceeds the schedulable "
                f"budget of {self.scheduler.max_num_batched_tokens} tokens "
                f"(max_num_batched_tokens); it can never be scheduled."
            )
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
        if not seqs:
            sched = self.scheduler
            # An empty decode batch after preemption is recoverable: the
            # preempted sequences go back to waiting and prefill reschedules
            # them next step. Only truly dead states raise.
            if sched.prefill_only_mode and sched.running:
                raise RuntimeError(
                    "Prefill-only engine has unfinished sequences but never "
                    "decodes; use max_tokens=1 (generate_single_token) or "
                    "construct the engine without prefill_only_mode / with a "
                    "larger max_tokens_hint."
                )
            head = sched.waiting[0] if sched.waiting else None
            if (
                head is not None
                and not sched.running
                and (
                    len(head) > sched.max_num_batched_tokens
                    or not sched.block_manager.can_allocate(head)
                )
            ):
                raise RuntimeError(
                    "Request can never be scheduled: the waiting head needs "
                    f"{len(head)} tokens (budget "
                    f"{sched.max_num_batched_tokens}) and "
                    f"{head.num_blocks} KV blocks (free "
                    f"{len(sched.block_manager.free_block_ids)}). "
                    "Lower max_model_len or raise max_num_batched_tokens / "
                    "gpu_memory_utilization."
                )
            # Nothing runnable this step; the caller loops again once the
            # scheduler frees blocks or new requests arrive.
            return [], 0
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
        else:
            _check_batch_lengths(
                {"prompts": len(prompts), "sampling_params": len(sampling_params)}
            )
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

    def _prepare_multimodal_batch(self, requests: list[dict], processor):
        """Shared multimodal preprocessing for generation and logits APIs.

        Extracts text/images from requests, runs the processor batch, and
        unbatches the results back per request. Pure extraction from the
        former generate_multimodal body; behaviour is unchanged.
        """
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

        return (
            all_texts,
            per_request_input_ids,
            per_request_pixel_values,
            per_request_image_grid_thw,
        )

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
        else:
            _check_batch_lengths(
                {"requests": len(requests), "sampling_params": len(sampling_params)}
            )

        (
            _,
            per_request_input_ids,
            per_request_pixel_values,
            per_request_image_grid_thw,
        ) = self._prepare_multimodal_batch(requests, processor)

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
        with self._left_padding():
            return self._embed_batch_impl(texts, images, use_tqdm)

    def _embed_batch_impl(
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
            
            # Qwen3-VL embedding models follow the official script
            # (scripts/qwen3_vl_embedding.py): system instruction +
            # add_generation_prompt=True, last-token pooling, no anchor.
            # NOTE: do NOT append the endoftext anchor manually: the Qwen
            # embedding tokenizers auto-append it via add_special_tokens,
            # so a manual append doubles it and shifts pooling.
            embed_type = getattr(self.model_runner.config, "embedding_type", None)
            is_qwen3_vl_embed = embed_type == "qwen3_vl"

            for i, text in enumerate(texts):
                if images and i < len(images) and images[i] is not None:
                    img = images[i] if isinstance(images[i], list) else [images[i]]
                    messages_batch.append([
                        *(
                            [{
                                "role": "system",
                                "content": [{"type": "text", "text": "Represent the user's input."}],
                            }]
                            if is_qwen3_vl_embed else []
                        ),
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
                        *(
                            [{
                                "role": "system",
                                "content": [{"type": "text", "text": "Represent the user's input."}],
                            }]
                            if is_qwen3_vl_embed else []
                        ),
                        {"role": "user", "content": [{"type": "text", "text": text}]}
                    ])
                    all_images_list.append(None)
                    text_only_fallback_indices.append(i)

            # Batch apply_chat_template (much faster than serial calls)
            try:
                all_formatted_texts = processor.apply_chat_template(
                    messages_batch, tokenize=False,
                    add_generation_prompt=is_qwen3_vl_embed,
                )
            except (AttributeError, TypeError):
                # Fallback to serial if batch not supported
                all_formatted_texts = []
                for msgs in messages_batch:
                    all_formatted_texts.append(
                        processor.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=is_qwen3_vl_embed)
                    )

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

                for idx, orig_idx in enumerate(multimodal_indices):
                    input_ids_list[orig_idx] = multimodal_input_ids[idx]
                    if multimodal_attention_mask is not None:
                        attention_mask_list[orig_idx] = multimodal_attention_mask[idx]

                for idx, orig_idx in enumerate(text_only_indices):
                    input_ids_list[orig_idx] = text_only_input_ids[idx]
                    if text_only_attention_mask is not None:
                        attention_mask_list[orig_idx] = text_only_attention_mask[idx]

                input_ids_tensor = torch.stack(input_ids_list)
                if attention_mask_list[0] is not None:
                    attention_mask_tensor = torch.stack(attention_mask_list)
                else:
                    attention_mask_tensor = None

                # pixel_values / image_grid_thw from the processor are already
                # concatenated over the multimodal requests (in order). They are
                # NOT per-sequence rows: indexing pixel_values[idx] would grab a
                # single patch row, not a whole image. The model runner assigns
                # images to sequences by scanning placeholder tokens, so the
                # concatenated tensors keep the correct mapping.
                pixel_values_batch = self._normalize_pixel_values(
                    multimodal_pixel_values, len(multimodal_texts)
                )
                image_grid_thw_batch = multimodal_image_grid_thw
            elif multimodal_input_ids is not None:
                input_ids_tensor = multimodal_input_ids
                attention_mask_tensor = multimodal_attention_mask
                pixel_values_batch = self._normalize_pixel_values(
                    multimodal_pixel_values, len(multimodal_texts)
                )
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
            
            # Qwen3-Embedding tokenizers auto-append the endoftext
            # token via add_special_tokens, matching the official model
            # card usage; do not append it manually (it would double).
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
        with self._left_padding():
            return self._rerank_batch_impl(query_doc_pairs, images, use_tqdm)

    def _rerank_batch_impl(
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


        batch_size = len(query_doc_pairs)

        # Multimodal reranking needs a processor; swallowing a load failure here
        # would silently downgrade to text-only scoring, so let it surface.
        processor = None
        if images:
            processor = self._get_processor()
            if hasattr(processor, "tokenizer"):
                processor.tokenizer.padding_side = "left"

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
            elif reranker_type == "jina_v3":
                # jina-reranker-v3 locates query/doc embeddings at special
                # tokens; the generic sep format omits them, which collapses
                # every score. Build its listwise prompt (one passage per pair).
                prefix = (
                    "<|im_start|>system\n"
                    "You are a search relevance expert who can determine a ranking "
                    "of the passages based on how relevant they are to the query. "
                    "If the query is a question, how relevant a passage is depends "
                    "on how well it answers the question. If not, try to analyze "
                    "the intent of the query and assess how well each passage "
                    "satisfies the intent. If an instruction is provided, you "
                    "should follow the instruction when determining the ranking."
                    "<|im_end|>\n<|im_start|>user\n"
                )
                suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                for query, doc in query_doc_pairs:
                    q = self.tokenizer.decode(query) if isinstance(query, list) else query
                    d = self.tokenizer.decode(doc) if isinstance(doc, list) else doc
                    body = (
                        f"I will provide you with 1 passages, each indicated by a "
                        f"numerical identifier. Rank the passages based on their "
                        f"relevance to query: {q}\n"
                        f'<passage id="0">\n{d}<|embed_token|>\n</passage>\n'
                        f"<query>\n{q}<|rerank_token|>\n</query>"
                    )
                    all_texts.append(prefix + body + suffix)
            else:
                for query, doc in query_doc_pairs:
                    if isinstance(query, str) and isinstance(doc, str):
                        sep_token = (
                            self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
                        )
                        text = f"{query}{self.tokenizer.decode([sep_token])}{doc}"
                        all_texts.append(text)
                    elif isinstance(query, list) and isinstance(doc, list):
                        sep_token = (
                            self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
                        )
                        combined_ids = query + [sep_token] + doc
                        all_texts.append(self.tokenizer.decode(combined_ids))
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
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        images: list | None = None,
        use_tqdm: bool = False,
    ) -> list[int]:
        """Prefill plus one sampling step, returning a token id per prompt.

        Args:
            prompts: Text prompts or pre-tokenized ids.
            sampling_params: Sampling settings; max_tokens is forced to 1.
            images: Optional images, one entry per prompt, for multimodal input.
            use_tqdm: Whether to show a progress bar.
        """
        if sampling_params is None:
            base_params = [SamplingParams(max_tokens=1)] * len(prompts)
        elif isinstance(sampling_params, list):
            _check_batch_lengths(
                {"prompts": len(prompts), "sampling_params": len(sampling_params)}
            )
            base_params = sampling_params
        else:
            base_params = [sampling_params] * len(prompts)

        # Never mutate the caller's objects; derive single-token copies instead.
        params = [replace(sp, max_tokens=1) for sp in base_params]

        if images:
            _check_batch_lengths(
                {"prompts": len(prompts), "images": len(images)}
            )
            # Images must go through the processor to become vision tokens;
            # add_request cannot do that on its own.
            requests = []
            for prompt, image in zip(prompts, images):
                text = (
                    prompt
                    if isinstance(prompt, str)
                    else self.tokenizer.decode(prompt)
                )
                image_list = image if isinstance(image, (list, tuple)) else [image]
                requests.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    *(
                                        {"type": "image", "image": img}
                                        for img in image_list
                                    ),
                                    {"type": "text", "text": text},
                                ],
                            }
                        ],
                        "images": list(image_list),
                    }
                )
            results = self.generate_multimodal(
                requests, params, self._get_processor(), use_tqdm=use_tqdm
            )
            return [r["token_ids"][0] if r["token_ids"] else None for r in results]

        if use_tqdm:
            pbar = tqdm(
                total=len(prompts), desc="Single Token", dynamic_ncols=True
            )

        for prompt, sp in zip(prompts, params):
            self.add_request(prompt, sp)

        outputs = {}
        while not self.is_finished():
            output, _ = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids[0] if token_ids else None
                if use_tqdm:
                    pbar.update(1)

        if use_tqdm:
            pbar.close()

        return [outputs[seq_id] for seq_id in sorted(outputs)]

    # ------------------------------------------------------------------
    # Deterministic last-position logits API (no sampling, no scheduler).
    # ------------------------------------------------------------------

    _last_logits_seq_counter = itertools.count(1_000_000_000)

    def _next_last_logits_seq_ids(self, count: int) -> list[int]:
        """Fresh sequence ids for one last_logits call; far from the
        scheduler's small Sequence ids so GDN state slots never collide."""
        return [next(self._last_logits_seq_counter) for _ in range(count)]

    def _vocab_size(self) -> int:
        """Vocabulary size from the HF config (text_config for multimodal)."""
        hf_config = self.model_runner.config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        return int(text_config.vocab_size)

    @staticmethod
    def _validate_candidates(candidate_token_ids, vocab_size: int) -> None:
        if candidate_token_ids is None:
            raise ValueError(
                "candidate_token_ids is required; pass explicit candidate "
                "tokens instead of relying on full-vocabulary logits"
            )
        if not isinstance(candidate_token_ids, (list, tuple)):
            raise TypeError("candidate_token_ids must be a list of token id lists")
        for row in candidate_token_ids:
            if not isinstance(row, (list, tuple)) or not row:
                raise ValueError(
                    "Each request needs a nonempty candidate_token_ids list"
                )
            for token in row:
                if not isinstance(token, int) or not 0 <= token < vocab_size:
                    raise ValueError(
                        f"Candidate token {token!r} is out of the vocabulary "
                        f"range [0, {vocab_size})"
                    )

    def prefill_last_logits(
        self,
        prompts: list[str] | list[list[int]],
        candidate_token_ids: list[list[int]],
    ) -> dict:
        """One batched prefill; return last-position candidate logits.

        Deterministic: no sampling, one logit row per prompt, result order
        equals input order. String prompts are encoded with
        ``add_special_tokens=False``; pre-tokenized id lists skip the
        tokenizer entirely (SemIf-style callers should pass ids to keep
        parity with their own encoding).
        """
        if not prompts:
            raise ValueError("prompts must be nonempty")
        if len(prompts) != len(candidate_token_ids):
            raise ValueError(
                f"prompts ({len(prompts)}) and candidate_token_ids "
                f"({len(candidate_token_ids)}) must have equal length"
            )
        vocab_size = self._vocab_size()
        self._validate_candidates(candidate_token_ids, vocab_size)

        encoded = []
        for prompt in prompts:
            if isinstance(prompt, str):
                # No chat template, no special tokens: the caller owns the
                # exact prompt text (parity with Transformers encoding).
                encoded.append(self.tokenizer.encode(prompt, add_special_tokens=False))
            elif isinstance(prompt, (list, tuple)):
                encoded.append(list(prompt))
            else:
                raise TypeError("Each prompt must be a string or a token id list")
            if not encoded[-1]:
                raise ValueError("A prompt is empty after tokenization")

        max_len = max(len(ids) for ids in encoded)
        max_cand = max(len(cands) for cands in candidate_token_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.int64)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.int64)
        candidate_mask = torch.zeros(
            (len(candidate_token_ids), max_cand), dtype=torch.bool
        )
        candidate_ids = torch.zeros(
            (len(candidate_token_ids), max_cand), dtype=torch.int64
        )
        for i, (ids, cands) in enumerate(zip(encoded, candidate_token_ids)):
            # Left padding: real tokens at the end, pad at the start.
            input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.int64)
            attention_mask[i, max_len - len(ids):] = 1
            candidate_ids[i, : len(cands)] = torch.tensor(cands, dtype=torch.int64)
            candidate_mask[i, : len(cands)] = True

        seq_ids = self._next_last_logits_seq_ids(len(encoded))
        gathered = self.model_runner.call(
            "last_logits",
            input_ids,
            None,  # positions: last_logits derives per-sequence positions
            attention_mask,
            candidate_ids,
            seq_ids,
        )

        return {
            "logits": gathered,
            "candidate_mask": candidate_mask,
            "candidate_token_ids": [list(c) for c in candidate_token_ids],
            "sequence_lengths": [len(ids) for ids in encoded],
            "model": self.config.model,
            "revision": self.config.model_revision,
            "dtype": str(next(self.model_runner.model.parameters()).dtype),
            "backend": "nanovllm",
        }

    def prefill_last_logits_multimodal(
        self,
        requests: list[dict],
        candidate_token_ids: list[list[int]],
    ) -> dict:
        """Batched prefill over mixed multimodal/text requests; last-position
        candidate logits, no sampling.

        Each request is a dict in one of two shapes:

        - ``{"messages": [...], "images": [PIL.Image, ...]}`` (or
          ``{"text": ..., "images": [...]}``): processed through the model's
          processor; vision placeholders are expanded exactly like
          ``generate_multimodal``.
        - ``{"input_ids": [...]}``: pre-tokenized text; the tokenizer is
          bypassed so the caller owns the exact encoding.

        Result order equals request order. Returned contract matches
        ``prefill_last_logits``.
        """
        if not requests:
            raise ValueError("requests must be nonempty")
        if len(requests) != len(candidate_token_ids):
            raise ValueError(
                f"requests ({len(requests)}) and candidate_token_ids "
                f"({len(candidate_token_ids)}) must have equal length"
            )
        vocab_size = self._vocab_size()
        self._validate_candidates(candidate_token_ids, vocab_size)

        encoded: list[list[int] | None] = [None] * len(requests)
        per_request_pixel_values: list = [None] * len(requests)
        per_request_image_grid_thw: list = [None] * len(requests)

        mm_requests = []
        mm_positions = []
        for i, request in enumerate(requests):
            if not isinstance(request, dict):
                raise TypeError("Each request must be a dict")
            if "input_ids" in request:
                ids = list(request["input_ids"])
                if not ids:
                    raise ValueError("A request has empty input_ids")
                encoded[i] = ids
            elif "messages" in request or "text" in request or "images" in request:
                mm_requests.append(request)
                mm_positions.append(i)
            else:
                raise ValueError(
                    "Each request must contain 'input_ids' or "
                    "'messages'/'text'/'images'"
                )

        if mm_requests:
            processor = self._get_processor()
            (
                _,
                mm_input_ids,
                mm_pixel_values,
                mm_image_grid_thw,
            ) = self._prepare_multimodal_batch(mm_requests, processor)
            for j, i in enumerate(mm_positions):
                ids = mm_input_ids[j]
                if not ids:
                    raise ValueError("A multimodal request produced empty input_ids")
                thw = mm_image_grid_thw[j]
                if thw is not None:
                    thw_2d = thw.squeeze(0) if thw.dim() == 3 else thw
                    expanded, _, _ = self._expand_vision_placeholders(ids, thw_2d)
                    ids = expanded
                encoded[i] = ids
                per_request_pixel_values[i] = mm_pixel_values[j]
                per_request_image_grid_thw[i] = thw

        # Merge per-request vision tensors in request order. Qwen-VL style
        # pixel_values are [total_patches, hidden]; concatenating along dim 0
        # reproduces the processor's original batch layout, which is what
        # _build_vision_placeholders expects (it assigns images to sequences
        # by scanning placeholder tokens).
        pixel_values_batch = None
        pv_parts = [pv for pv in per_request_pixel_values if pv is not None]
        if pv_parts:
            if not all(isinstance(pv, torch.Tensor) for pv in pv_parts):
                raise NotImplementedError(
                    "list-style pixel_values (e.g. LlavaNext) are not "
                    "supported by prefill_last_logits_multimodal"
                )
            pixel_values_batch = torch.cat(pv_parts, dim=0)
        image_grid_thw_batch = None
        thw_parts = [
            (t.squeeze(0) if t.dim() == 3 else t)
            for t in per_request_image_grid_thw
            if t is not None
        ]
        if thw_parts:
            image_grid_thw_batch = torch.cat(thw_parts, dim=0)

        # Left-padded batch assembly, identical to prefill_last_logits.
        max_len = max(len(ids) for ids in encoded)
        max_cand = max(len(cands) for cands in candidate_token_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        input_ids = torch.full((len(requests), max_len), pad_id, dtype=torch.int64)
        attention_mask = torch.zeros((len(requests), max_len), dtype=torch.int64)
        candidate_mask = torch.zeros(
            (len(candidate_token_ids), max_cand), dtype=torch.bool
        )
        candidate_ids = torch.zeros(
            (len(candidate_token_ids), max_cand), dtype=torch.int64
        )
        for i, (ids, cands) in enumerate(zip(encoded, candidate_token_ids)):
            # Left padding: real tokens at the end, pad at the start.
            input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.int64)
            attention_mask[i, max_len - len(ids):] = 1
            candidate_ids[i, : len(cands)] = torch.tensor(cands, dtype=torch.int64)
            candidate_mask[i, : len(cands)] = True

        seq_ids = self._next_last_logits_seq_ids(len(requests))
        gathered = self.model_runner.call(
            "last_logits",
            input_ids,
            None,  # positions: last_logits derives per-sequence positions
            attention_mask,
            candidate_ids,
            seq_ids,
            pixel_values=pixel_values_batch,
            image_grid_thw=image_grid_thw_batch,
        )

        return {
            "logits": gathered,
            "candidate_mask": candidate_mask,
            "candidate_token_ids": [list(c) for c in candidate_token_ids],
            "sequence_lengths": [len(ids) for ids in encoded],
            "model": self.config.model,
            "revision": self.config.model_revision,
            "dtype": str(next(self.model_runner.model.parameters()).dtype),
            "backend": "nanovllm",
        }