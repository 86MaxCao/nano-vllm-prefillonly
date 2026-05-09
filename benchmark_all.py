"""
Comprehensive benchmark: Prefill-Only vs Normal inference.

For each model, compares three modes:
  1. HuggingFace Transformers (ground truth)
  2. nano-vllm Prefill-Only (optimized, no KV cache)
  3. nano-vllm Normal (with KV cache)

Metrics per model:
  - Logits / output consistency (cosine sim, max diff, exact match)
  - Speed (mean / median / p90 latency)
  - VRAM (peak memory MB)

Results are written to BENCHMARK_RESULTS.md

Usage:
    python benchmark_all.py                          # all models
    python benchmark_all.py --category text_gen      # specific category
    python benchmark_all.py --model qwen3-0.6b       # specific model
    python benchmark_all.py --gpu 0                  # specific GPU
    python benchmark_all.py --skip-missing            # skip unavailable models
"""
import argparse
import gc
import io
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request

# ── Suppress DEBUG prints from nanovllm internals ────────────────────────────
class _DebugFilter:
    """Filter stdout to suppress lines containing [DEBUG or other noisy patterns."""
    NOISY = ("[DEBUG", "logits shape", "logits dtype", "logits min", "logits mean",
             "logits has", "logits all", "logits[", "token_ids:", "x shape",
             "x dtype", "x min", "x has", "x all", "x[", "tp_size:",
             "temperatures", "is_prefill", "sequence_lengths", "num_seqs",
             "first seq", "last_indices", "Expected last", "last_indices match",
             "self.weight shape", "self.weight dtype", "x mean:",
             "context.cu_seqlens_q", "input x mean", "logits device",
             "[Memory] Cleaned", "[KV Cache]", "Loading weights from")

    def __init__(self, stream):
        self.stream = stream
        self._buf = ""

    def write(self, text):
        # Only filter complete lines
        if '\n' not in text:
            self._buf += text
            return len(text)
        lines = (self._buf + text).split('\n')
        self._buf = lines.pop()  # last incomplete part
        for line in lines:
            if not any(p in line for p in self.NOISY):
                self.stream.write(line + '\n')
        return len(text)

    def flush(self):
        if self._buf:
            if not any(p in self._buf for p in self.NOISY):
                self.stream.write(self._buf)
            self._buf = ""
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)

sys.stdout = _DebugFilter(sys.stdout)

import numpy as np
import torch

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub/hf_cache")

# ── Model configs ──────────────────────────────────────────────────────────────
# Each entry: key -> {category, path, llm_kwargs, hf_class, ...}
MODEL_CONFIGS = {
    # ── Text Generation ────────────────────────────────────────────────────
    "qwen3-0.6b": {
        "category": "text_gen",
        "path": f"{HF_CACHE}/Qwen3-0.6B",
        "llm_kwargs": {"prefill_only_mode": True, "single_token_mode": True},
        "llm_kwargs_normal": {"prefill_only_mode": False, "single_token_mode": True},
        "hf_class": "AutoModelForCausalLM",
    },
    # ── VL Generation ──────────────────────────────────────────────────────
    "qwen3-vl-2b": {
        "category": "vl_gen",
        "path": f"{HF_CACHE}/Qwen3-VL-2B-Instruct",
        "llm_kwargs": {
            "is_multimodal": True,
            "multimodal_model_type": "qwen3_vl",
            "prefill_only_mode": True,
            "single_token_mode": True,
            "trust_remote_code": True,
        },
        "llm_kwargs_normal": {
            "is_multimodal": True,
            "multimodal_model_type": "qwen3_vl",
            "prefill_only_mode": False,
            "single_token_mode": True,
            "trust_remote_code": True,
        },
        "hf_class": "Qwen3VLForConditionalGeneration",
        "model_type": "qwen3_vl",
    },
    "qwen2-vl-2b": {
        "category": "vl_gen",
        "path": f"{HF_CACHE}/Qwen2-VL-2B-Instruct",
        "llm_kwargs": {
            "is_multimodal": True,
            "multimodal_model_type": "qwen2_vl",
            "prefill_only_mode": True,
            "single_token_mode": True,
            "trust_remote_code": True,
        },
        "llm_kwargs_normal": {
            "is_multimodal": True,
            "multimodal_model_type": "qwen2_vl",
            "prefill_only_mode": False,
            "single_token_mode": True,
            "trust_remote_code": True,
        },
        "hf_class": "Qwen2VLForConditionalGeneration",
        "model_type": "qwen2_vl",
    },
    "qwen2.5-vl-3b": {
        "category": "vl_gen",
        "path": f"{HF_CACHE}/Qwen2.5-VL-3B-Instruct",
        "llm_kwargs": {
            "is_multimodal": True,
            "multimodal_model_type": "qwen2_5_vl",
            "prefill_only_mode": True,
            "single_token_mode": True,
            "trust_remote_code": True,
        },
        "llm_kwargs_normal": {
            "is_multimodal": True,
            "multimodal_model_type": "qwen2_5_vl",
            "prefill_only_mode": False,
            "single_token_mode": True,
            "trust_remote_code": True,
        },
        "hf_class": "Qwen2_5_VLForConditionalGeneration",
        "model_type": "qwen2_5_vl",
    },
    "llavanext-7b": {
        "category": "vl_gen",
        "path": f"{HF_CACHE}/llava-v1.6-mistral-7b-hf",
        "llm_kwargs": {
            "is_multimodal": True,
            "multimodal_model_type": "llavanext",
            "prefill_only_mode": True,
            "single_token_mode": True,
            "trust_remote_code": True,
        },
        "llm_kwargs_normal": {
            "is_multimodal": True,
            "multimodal_model_type": "llavanext",
            "prefill_only_mode": False,
            "single_token_mode": True,
            "trust_remote_code": True,
        },
        "hf_class": "LlavaNextForConditionalGeneration",
        "model_type": "llavanext",
    },
    # ── Text Embedding ─────────────────────────────────────────────────────
    "qwen3-embedding-0.6b": {
        "category": "text_embed",
        "path": f"{HF_CACHE}/Qwen3-Embedding-0.6B",
        "llm_kwargs": {
            "is_embedding": True,
            "embedding_type": "qwen3",
            "pooling_type": "LAST",
            "normalize_embeddings": True,
            "prefill_only_mode": True,
        },
        "llm_kwargs_normal": {
            "is_embedding": True,
            "embedding_type": "qwen3",
            "pooling_type": "LAST",
            "normalize_embeddings": True,
            "prefill_only_mode": False,
        },
    },
    "bge-gemma2": {
        "category": "text_embed",
        "path": f"{HF_CACHE}/bge-multilingual-gemma2",
        "llm_kwargs": {
            "is_embedding": True,
            "embedding_type": "gemma2",
            "pooling_type": "LAST",
            "normalize_embeddings": True,
            "prefill_only_mode": True,
        },
        "llm_kwargs_normal": {
            "is_embedding": True,
            "embedding_type": "gemma2",
            "pooling_type": "LAST",
            "normalize_embeddings": True,
            "prefill_only_mode": False,
        },
    },
    "jina-embed-v4": {
        "category": "text_embed",
        "path": f"{HF_CACHE}/jina-embeddings-v4",
        "llm_kwargs": {
            "is_embedding": True,
            "embedding_type": "jina_v4",
            "pooling_type": "LAST",
            "normalize_embeddings": True,
            "prefill_only_mode": True,
            "trust_remote_code": True,
        },
        "llm_kwargs_normal": {
            "is_embedding": True,
            "embedding_type": "jina_v4",
            "pooling_type": "LAST",
            "normalize_embeddings": True,
            "prefill_only_mode": False,
            "trust_remote_code": True,
        },
    },
    # ── VL Embedding ───────────────────────────────────────────────────────
    "qwen3-vl-embedding-2b": {
        "category": "vl_embed",
        "path": f"{HF_CACHE}/Qwen3-VL-Embedding-2B",
        "llm_kwargs": {
            "is_embedding": True,
            "embedding_type": "qwen3_vl",
            "is_multimodal": True,
            "multimodal_model_type": "qwen3_vl",
            "pooling_type": "LAST",
            "normalize_embeddings": True,
            "prefill_only_mode": True,
            "trust_remote_code": True,
        },
        "llm_kwargs_normal": {
            "is_embedding": True,
            "embedding_type": "qwen3_vl",
            "is_multimodal": True,
            "multimodal_model_type": "qwen3_vl",
            "pooling_type": "LAST",
            "normalize_embeddings": True,
            "prefill_only_mode": False,
            "trust_remote_code": True,
        },
    },
    "gme-qwen2-vl-2b": {
        "category": "vl_embed",
        "path": f"{HF_CACHE}/gme-Qwen2-VL-2B-Instruct",
        "llm_kwargs": {
            "is_embedding": True,
            "embedding_type": "qwen2_vl_gme",
            "is_multimodal": True,
            "multimodal_model_type": "qwen2_vl",
            "pooling_type": "LAST",
            "normalize_embeddings": True,
            "prefill_only_mode": True,
            "trust_remote_code": True,
        },
        "llm_kwargs_normal": {
            "is_embedding": True,
            "embedding_type": "qwen2_vl_gme",
            "is_multimodal": True,
            "multimodal_model_type": "qwen2_vl",
            "pooling_type": "LAST",
            "normalize_embeddings": True,
            "prefill_only_mode": False,
            "trust_remote_code": True,
        },
    },
    # ── Text Reranker ──────────────────────────────────────────────────────
    "qwen3-reranker-0.6b": {
        "category": "text_rerank",
        "path": f"{HF_CACHE}/Qwen3-Reranker-0.6B",
        "llm_kwargs": {
            "is_reranker": True,
            "reranker_type": "qwen3",
            "is_original_qwen3_reranker": True,
            "classifier_from_token": ["no", "yes"],
            "prefill_only_mode": True,
        },
        "llm_kwargs_normal": {
            "is_reranker": True,
            "reranker_type": "qwen3",
            "is_original_qwen3_reranker": True,
            "classifier_from_token": ["no", "yes"],
            "prefill_only_mode": False,
        },
    },
    "jina-reranker-v3": {
        "category": "text_rerank",
        "path": f"{HF_CACHE}/jina-reranker-v3",
        "llm_kwargs": {
            "is_reranker": True,
            "reranker_type": "jina_v3",
            "prefill_only_mode": True,
        },
        "llm_kwargs_normal": {
            "is_reranker": True,
            "reranker_type": "jina_v3",
            "prefill_only_mode": False,
        },
    },
    "bge-reranker-gemma": {
        "category": "text_rerank",
        "path": f"{HF_CACHE}/bge-reranker-v2-gemma",
        "llm_kwargs": {
            "is_reranker": True,
            "reranker_type": "gemma",
            "is_original_qwen3_reranker": True,
            "classifier_from_token": ["Yes"],
            "prefill_only_mode": True,
        },
        "llm_kwargs_normal": {
            "is_reranker": True,
            "reranker_type": "gemma",
            "is_original_qwen3_reranker": True,
            "classifier_from_token": ["Yes"],
            "prefill_only_mode": False,
        },
    },
    # ── VL Reranker ────────────────────────────────────────────────────────
    "qwen3-vl-reranker-2b": {
        "category": "vl_rerank",
        "path": f"{HF_CACHE}/Qwen3-VL-Reranker-2B",
        "llm_kwargs": {
            "is_reranker": True,
            "reranker_type": "qwen3_vl",
            "is_multimodal": True,
            "multimodal_model_type": "qwen3_vl",
            "is_original_qwen3_reranker": True,
            "classifier_from_token": ["no", "yes"],
            "prefill_only_mode": True,
            "trust_remote_code": True,
        },
        "llm_kwargs_normal": {
            "is_reranker": True,
            "reranker_type": "qwen3_vl",
            "is_multimodal": True,
            "multimodal_model_type": "qwen3_vl",
            "is_original_qwen3_reranker": True,
            "classifier_from_token": ["no", "yes"],
            "prefill_only_mode": False,
            "trust_remote_code": True,
        },
    },
    "jina-reranker-m0": {
        "category": "vl_rerank",
        "path": f"{HF_CACHE}/jina-reranker-m0",
        "llm_kwargs": {
            "is_reranker": True,
            "reranker_type": "jina_m0",
            "is_multimodal": True,
            "multimodal_model_type": "qwen2_vl",
            "prefill_only_mode": True,
            "trust_remote_code": True,
        },
        "llm_kwargs_normal": {
            "is_reranker": True,
            "reranker_type": "jina_m0",
            "is_multimodal": True,
            "multimodal_model_type": "qwen2_vl",
            "prefill_only_mode": False,
            "trust_remote_code": True,
        },
    },
}

CATEGORIES = [
    "text_gen", "vl_gen",
    "text_embed", "vl_embed",
    "text_rerank", "vl_rerank",
]

# ── Helper functions ────────────────────────────────────────────────────────────

def download_image(url):
    with urllib.request.urlopen(url) as resp:
        return __import__("PIL").Image.open(io.BytesIO(resp.read())).convert("RGB")


def get_peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0


def reset_memory_stats():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    # Set random MASTER_PORT to avoid EADDRINUSE between runs
    import random
    os.environ["MASTER_PORT"] = str(random.randint(30000, 60000))
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def cleanup_llm(llm):
    """Force cleanup of LLM object."""
    try:
        llm.exit()
    except Exception:
        pass
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


def set_random_master_port():
    """Set a random MASTER_PORT to avoid port conflicts."""
    import random
    port = random.randint(30000, 60000)
    os.environ["MASTER_PORT"] = str(port)
    # Also reset the distributed group so a new one is created
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


# ── Logits / output extraction from nano-vllm ────────────────────────────────

def get_nanovllm_logits(llm, prompts, is_multimodal=False, processor=None, images=None):
    """Extract logits from nano-vllm by hooking into model_runner.

    Returns (token_ids, logits_tensor, latency_list).
    """
    from nanovllm import SamplingParams

    captured = {}

    # Hook into model_runner.run to capture logits before sampling
    original_run = llm.model_runner.run

    def hooked_run(seqs, is_prefill):
        # Call original to get token_ids
        token_ids = original_run(seqs, is_prefill)
        # After run_model, the logits are stored in model_runner._last_logits
        # We need to capture them during run_model, not after
        return token_ids

    # Alternative: hook into run_model directly to capture logits
    original_run_model = llm.model_runner.run_model

    def hooked_run_model(input_ids, positions, is_prefill, **kwargs):
        logits = original_run_model(input_ids, positions, is_prefill, **kwargs)
        if logits is not None:
            captured["logits"] = logits.detach().cpu().float().clone()
        return logits

    llm.model_runner.run_model = hooked_run_model

    try:
        # Run inference
        torch.cuda.synchronize()
        start = time.perf_counter()

        if is_multimodal and processor and images:
            requests = []
            for prompt, img in zip(prompts, images):
                chat = processor.apply_chat_template(
                    [{"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ]}],
                    tokenize=False, add_generation_prompt=True,
                )
                requests.append({"text": chat, "images": [img]})
            outputs = llm.generate_multimodal(
                requests,
                SamplingParams(temperature=1e-3, max_tokens=1),
                processor, use_tqdm=False,
            )
            token_ids = []
            for out in outputs:
                if "token_ids" in out and out["token_ids"]:
                    tid = out["token_ids"][0] if isinstance(out["token_ids"], list) else out["token_ids"]
                    token_ids.append(tid)
                else:
                    token_ids.append(-1)
        else:
            token_ids = llm.generate_single_token(
                prompts,
                SamplingParams(temperature=1e-3, max_tokens=1),
                use_tqdm=False,
            )

        torch.cuda.synchronize()
        latency = time.perf_counter() - start
    finally:
        # Restore original method
        llm.model_runner.run_model = original_run_model

    logits = captured.get("logits", None)
    return token_ids, logits, latency


def get_nanovllm_embeddings(llm, texts, is_multimodal=False, images=None):
    """Extract embeddings from nano-vllm."""
    torch.cuda.synchronize()
    start = time.perf_counter()

    if is_multimodal and images:
        embeddings = llm.embed_batch(texts, images=images, use_tqdm=False)
    else:
        embeddings = llm.embed_batch(texts, use_tqdm=False)

    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    return embeddings.cpu().float(), latency


def get_nanovllm_scores(llm, query_doc_pairs, is_multimodal=False, images=None):
    """Extract reranker scores from nano-vllm."""
    torch.cuda.synchronize()
    start = time.perf_counter()

    if is_multimodal and images:
        result = llm.rerank_batch(query_doc_pairs, images=images, use_tqdm=False)
    else:
        result = llm.rerank_batch(query_doc_pairs, use_tqdm=False)

    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    # Handle both pointwise and listwise return
    if isinstance(result, tuple):
        scores = result[0].cpu().float()
    else:
        scores = result.cpu().float()

    return scores, latency


# ── HuggingFace Transformers baselines ────────────────────────────────────────

def hf_text_gen_logits(model_path, prompts, num_warmup=2, num_iter=10):
    """Get logits from HF model for text generation."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        trust_remote_code=True, attn_implementation="flash_attention_2",
    ).eval().cuda()

    inputs = tokenizer(prompts, padding=True, truncation=True,
                       return_tensors="pt", max_length=2048).to(model.device)

    # Get reference logits (last token position)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        if 'attention_mask' in inputs:
            seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
            last_logits = logits[torch.arange(logits.shape[0], device=logits.device), seq_lengths, :]
        else:
            last_logits = logits[:, -1, :]

        ref_logits = last_logits.cpu().float().clone()
        ref_token_ids = last_logits.argmax(dim=-1).cpu().tolist()

    # Warmup + benchmark
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(num_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    peak_mem = get_peak_mb()
    del model, tokenizer, inputs
    reset_memory_stats()

    return ref_logits, ref_token_ids, latencies, peak_mem


def hf_vl_gen_logits(model_path, prompts, images, model_type, num_warmup=2, num_iter=5):
    """Get logits from HF model for VL generation."""
    from transformers import AutoProcessor
    try:
        from transformers import (
            LlavaNextForConditionalGeneration,
            Qwen2VLForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
        )
    except ImportError:
        LlavaNextForConditionalGeneration = None
        Qwen2VLForConditionalGeneration = None
        Qwen2_5_VLForConditionalGeneration = None
        Qwen3VLForConditionalGeneration = None

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    cls_map = {
        "qwen3_vl": Qwen3VLForConditionalGeneration,
        "qwen2_vl": Qwen2VLForConditionalGeneration,
        "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
        "llavanext": LlavaNextForConditionalGeneration,
    }
    model_cls = cls_map.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = model_cls.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="flash_attention_2",
    ).eval().cuda()

    # Build requests
    requests_list = []
    for prompt, img in zip(prompts, images):
        chat = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}],
            tokenize=False, add_generation_prompt=True,
        )
        requests_list.append({"text": chat, "images": [img]})

    # Get reference logits one by one (different image sizes)
    ref_logits_list = []
    ref_token_ids = []
    with torch.no_grad():
        for req in requests_list:
            proc_out = processor(text=[req["text"]], images=req["images"],
                                 return_tensors="pt", padding=True)
            inp = {k: v.to(model.device) for k, v in proc_out.items()}
            out = model(**inp)
            if hasattr(out, 'logits'):
                logits = out.logits
            else:
                raise ValueError("Model output has no logits")
            last_logit = logits[:, -1, :]
            ref_logits_list.append(last_logit.cpu().float().clone())
            ref_token_ids.append(last_logit.argmax(dim=-1).item())

    ref_logits = torch.cat(ref_logits_list, dim=0)

    # Warmup + benchmark (first request only for speed estimate)
    req0 = requests_list[0]
    proc0 = processor(text=[req0["text"]], images=req0["images"],
                      return_tensors="pt", padding=True)
    inp0 = {k: v.to(model.device) for k, v in proc0.items()}

    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(**inp0)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(num_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**inp0)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    peak_mem = get_peak_mb()
    del model
    reset_memory_stats()

    return ref_logits, ref_token_ids, latencies, peak_mem


def hf_text_embed(model_path, texts, num_warmup=2, num_iter=10):
    """Get embeddings from HF model."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation="flash_attention_2",
    ).eval().cuda()

    inputs = tokenizer(texts, padding=True, truncation=True,
                       return_tensors="pt", max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Last token pooling
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0]
        if 'attention_mask' in inputs:
            seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
            embeddings = hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lengths, :]
        else:
            embeddings = hidden[:, -1, :]
        embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=-1)
        ref_embeddings = embeddings.cpu().clone()

    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(num_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    peak_mem = get_peak_mb()
    del model, tokenizer, inputs
    reset_memory_stats()

    return ref_embeddings, latencies, peak_mem


def hf_text_rerank(model_path, query_doc_pairs, num_warmup=2, num_iter=10):
    """Get reranker scores from HF model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        trust_remote_code=True, attn_implementation="flash_attention_2",
    ).eval().cuda()

    # Format query-doc pairs
    yes_token_id = tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    no_token_id = tokenizer("no", add_special_tokens=False)["input_ids"][0]

    texts = [f"{q} {d}" for q, d in query_doc_pairs]
    inputs = tokenizer(texts, padding=True, truncation=True,
                       return_tensors="pt", max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        if 'attention_mask' in inputs:
            seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
            last_logits = logits[torch.arange(logits.shape[0], device=logits.device), seq_lengths, :]
        else:
            last_logits = logits[:, -1, :]
        # yes - no logit difference -> sigmoid -> score
        yes_logits = last_logits[:, yes_token_id]
        no_logits = last_logits[:, no_token_id]
        scores = torch.sigmoid(yes_logits - no_logits).cpu().float().clone()

    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(num_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    peak_mem = get_peak_mb()
    del model, tokenizer, inputs
    reset_memory_stats()

    return scores, latencies, peak_mem


# ── Main benchmark functions ───────────────────────────────────────────────────

def bench_text_gen(name, cfg, gpu_id, num_warmup=3, num_iter=20):
    """Benchmark text generation model."""
    model_path = cfg["path"]
    if not os.path.exists(model_path):
        return None

    results = {"name": name, "category": cfg["category"], "model_path": model_path}

    # Test prompts
    prompts = [
        "Is the capital of China Beijing? Answer with Yes or No.",
        "Is 2+2 equal to 4? Answer with Yes or No.",
        "Is the Earth round? Answer with Yes or No.",
        "Is Python a programming language? Answer with Yes or No.",
        "Is water made of H2O? Answer with Yes or No.",
    ]

    # 1. HF baseline
    print(f"  [1/3] HuggingFace Transformers ...")
    reset_memory_stats()
    try:
        ref_logits, ref_token_ids, hf_latencies, hf_peak = hf_text_gen_logits(
            model_path, prompts, num_warmup=num_warmup, num_iter=num_iter)
        results["hf_ok"] = True
        results["hf_ref_logits_shape"] = list(ref_logits.shape)
        results["hf_ref_token_ids"] = ref_token_ids
        results["hf_latency_mean"] = float(np.mean(hf_latencies))
        results["hf_latency_median"] = float(np.median(hf_latencies))
        results["hf_latency_p90"] = float(np.percentile(hf_latencies, 90))
        results["hf_peak_mem_mb"] = hf_peak
    except Exception as e:
        print(f"  HF FAILED: {e}")
        results["hf_ok"] = False
        ref_logits = None

    # 2. Prefill-Only
    print(f"  [2/3] nano-vllm Prefill-Only ...")
    reset_memory_stats()
    set_random_master_port()
    try:
        from nanovllm import LLM, SamplingParams
        llm_kwargs = dict(cfg["llm_kwargs"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm_kwargs["seed"] = 0
        llm = LLM(model_path, **llm_kwargs)

        # Warmup
        for _ in range(num_warmup):
            llm.generate_single_token(prompts[:1], SamplingParams(temperature=1e-3, max_tokens=1), use_tqdm=False)

        # Get logits + token_ids
        po_token_ids, po_logits, po_single_latency = get_nanovllm_logits(llm, prompts)

        # Benchmark
        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate_single_token(prompts, SamplingParams(temperature=1e-3, max_tokens=1), use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        po_peak = get_peak_mb()
        results["po_ok"] = True
        results["po_token_ids"] = po_token_ids
        results["po_logits"] = po_logits
        results["po_latency_mean"] = float(np.mean(latencies))
        results["po_latency_median"] = float(np.median(latencies))
        results["po_latency_p90"] = float(np.percentile(latencies, 90))
        results["po_peak_mem_mb"] = po_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Prefill-Only FAILED: {e}")
        traceback.print_exc()
        results["po_ok"] = False
        po_logits = None

    # 3. Normal (with KV cache)
    print(f"  [3/3] nano-vllm Normal ...")
    reset_memory_stats()
    set_random_master_port()
    try:
        from nanovllm import LLM, SamplingParams
        llm_kwargs = dict(cfg["llm_kwargs_normal"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm_kwargs["seed"] = 0
        llm = LLM(model_path, **llm_kwargs)

        # Warmup
        for _ in range(num_warmup):
            llm.generate_single_token(prompts[:1], SamplingParams(temperature=1e-3, max_tokens=1), use_tqdm=False)

        # Get logits + token_ids
        nm_token_ids, nm_logits, nm_single_latency = get_nanovllm_logits(llm, prompts)

        # Benchmark
        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate_single_token(prompts, SamplingParams(temperature=1e-3, max_tokens=1), use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        nm_peak = get_peak_mb()
        results["nm_ok"] = True
        results["nm_token_ids"] = nm_token_ids
        results["nm_logits"] = nm_logits
        results["nm_latency_mean"] = float(np.mean(latencies))
        results["nm_latency_median"] = float(np.median(latencies))
        results["nm_latency_p90"] = float(np.percentile(latencies, 90))
        results["nm_peak_mem_mb"] = nm_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Normal FAILED: {e}")
        traceback.print_exc()
        results["nm_ok"] = False
        nm_logits = None

    # ── Compare logits ─────────────────────────────────────────────────────
    compare_logits_results(results, ref_logits, po_logits, nm_logits)

    return results


def bench_vl_gen(name, cfg, gpu_id, num_warmup=2, num_iter=5):
    """Benchmark VL generation model."""
    model_path = cfg["path"]
    if not os.path.exists(model_path):
        return None

    results = {"name": name, "category": cfg["category"], "model_path": model_path}

    # Download images
    image_urls = (
        "http://images.cocodataset.org/val2017/000000000285.jpg",
        "http://images.cocodataset.org/val2017/000000000632.jpg",
        "http://images.cocodataset.org/val2017/000000000724.jpg",
    )
    prompts = [
        "What animal is in the picture? Answer with one word.",
        "Was the image taken indoors or outdoors? Answer with indoors or outdoors.",
        "Was the image taken outdoors? Answer with yes or no.",
    ]

    print("  Downloading test images ...")
    images = [download_image(url) for url in image_urls[:len(prompts)]]

    # 1. HF baseline
    print(f"  [1/3] HuggingFace Transformers ...")
    reset_memory_stats()
    try:
        ref_logits, ref_token_ids, hf_latencies, hf_peak = hf_vl_gen_logits(
            model_path, prompts, images, cfg["model_type"],
            num_warmup=num_warmup, num_iter=num_iter)
        results["hf_ok"] = True
        results["hf_ref_token_ids"] = ref_token_ids
        results["hf_latency_mean"] = float(np.mean(hf_latencies))
        results["hf_latency_median"] = float(np.median(hf_latencies))
        results["hf_latency_p90"] = float(np.percentile(hf_latencies, 90))
        results["hf_peak_mem_mb"] = hf_peak
    except Exception as e:
        print(f"  HF FAILED: {e}")
        traceback.print_exc()
        results["hf_ok"] = False
        ref_logits = None

    # 2. Prefill-Only
    print(f"  [2/3] nano-vllm Prefill-Only ...")
    reset_memory_stats()
    try:
        from nanovllm import LLM, SamplingParams
        from transformers import AutoProcessor

        llm_kwargs = dict(cfg["llm_kwargs"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm_kwargs["seed"] = 0
        llm = LLM(model_path, **llm_kwargs)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Warmup
        for _ in range(num_warmup):
            req0 = [{"text": processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": images[0]}, {"type": "text", "text": prompts[0]}]}],
                tokenize=False, add_generation_prompt=True),
                "images": [images[0]]}]
            llm.generate_multimodal(req0, SamplingParams(temperature=1e-3, max_tokens=1), processor, use_tqdm=False)

        po_token_ids, po_logits, _ = get_nanovllm_logits(
            llm, prompts, is_multimodal=True, processor=processor, images=images)

        # Benchmark
        requests_list = []
        for prompt, img in zip(prompts, images):
            chat = processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}],
                tokenize=False, add_generation_prompt=True,
            )
            requests_list.append({"text": chat, "images": [img]})

        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate_multimodal(requests_list, SamplingParams(temperature=1e-3, max_tokens=1), processor, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        po_peak = get_peak_mb()
        results["po_ok"] = True
        results["po_token_ids"] = po_token_ids
        results["po_logits"] = po_logits
        results["po_latency_mean"] = float(np.mean(latencies))
        results["po_latency_median"] = float(np.median(latencies))
        results["po_latency_p90"] = float(np.percentile(latencies, 90))
        results["po_peak_mem_mb"] = po_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Prefill-Only FAILED: {e}")
        traceback.print_exc()
        results["po_ok"] = False
        po_logits = None

    # 3. Normal
    print(f"  [3/3] nano-vllm Normal ...")
    reset_memory_stats()
    try:
        from nanovllm import LLM, SamplingParams
        from transformers import AutoProcessor

        llm_kwargs = dict(cfg["llm_kwargs_normal"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm_kwargs["seed"] = 0
        llm = LLM(model_path, **llm_kwargs)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        nm_token_ids, nm_logits, _ = get_nanovllm_logits(
            llm, prompts, is_multimodal=True, processor=processor, images=images)

        requests_list = []
        for prompt, img in zip(prompts, images):
            chat = processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}],
                tokenize=False, add_generation_prompt=True,
            )
            requests_list.append({"text": chat, "images": [img]})

        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate_multimodal(requests_list, SamplingParams(temperature=1e-3, max_tokens=1), processor, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        nm_peak = get_peak_mb()
        results["nm_ok"] = True
        results["nm_token_ids"] = nm_token_ids
        results["nm_logits"] = nm_logits
        results["nm_latency_mean"] = float(np.mean(latencies))
        results["nm_latency_median"] = float(np.median(latencies))
        results["nm_latency_p90"] = float(np.percentile(latencies, 90))
        results["nm_peak_mem_mb"] = nm_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Normal FAILED: {e}")
        traceback.print_exc()
        results["nm_ok"] = False
        nm_logits = None

    compare_logits_results(results, ref_logits,
                           results.get("po_logits"), results.get("nm_logits"))
    return results


def bench_text_embed(name, cfg, gpu_id, num_warmup=3, num_iter=20):
    """Benchmark text embedding model."""
    model_path = cfg["path"]
    if not os.path.exists(model_path):
        return None

    results = {"name": name, "category": cfg["category"], "model_path": model_path}
    texts = [
        "What is the capital of China?",
        "The capital of China is Beijing.",
        "Explain gravity",
        "Gravity is a force that attracts two bodies.",
        "Machine learning is a subset of AI.",
    ]

    # 1. HF baseline
    print(f"  [1/3] HuggingFace Transformers ...")
    reset_memory_stats()
    try:
        ref_embs, hf_latencies, hf_peak = hf_text_embed(model_path, texts, num_warmup, num_iter)
        results["hf_ok"] = True
        results["hf_latency_mean"] = float(np.mean(hf_latencies))
        results["hf_latency_median"] = float(np.median(hf_latencies))
        results["hf_latency_p90"] = float(np.percentile(hf_latencies, 90))
        results["hf_peak_mem_mb"] = hf_peak
        results["ref_embs"] = ref_embs
    except Exception as e:
        print(f"  HF FAILED: {e}")
        traceback.print_exc()
        results["hf_ok"] = False
        ref_embs = None

    # 2. Prefill-Only
    print(f"  [2/3] nano-vllm Prefill-Only ...")
    reset_memory_stats()
    try:
        from nanovllm import LLM
        llm_kwargs = dict(cfg["llm_kwargs"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm = LLM(model_path, **llm_kwargs)

        # Warmup
        for _ in range(num_warmup):
            llm.embed_batch(texts[:1], use_tqdm=False)

        po_embs, _ = get_nanovllm_embeddings(llm, texts)

        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.embed_batch(texts, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        po_peak = get_peak_mb()
        results["po_ok"] = True
        results["po_embs"] = po_embs
        results["po_latency_mean"] = float(np.mean(latencies))
        results["po_latency_median"] = float(np.median(latencies))
        results["po_latency_p90"] = float(np.percentile(latencies, 90))
        results["po_peak_mem_mb"] = po_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Prefill-Only FAILED: {e}")
        traceback.print_exc()
        results["po_ok"] = False

    # 3. Normal
    print(f"  [3/3] nano-vllm Normal ...")
    reset_memory_stats()
    try:
        from nanovllm import LLM
        llm_kwargs = dict(cfg["llm_kwargs_normal"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm = LLM(model_path, **llm_kwargs)

        nm_embs, _ = get_nanovllm_embeddings(llm, texts)

        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.embed_batch(texts, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        nm_peak = get_peak_mb()
        results["nm_ok"] = True
        results["nm_embs"] = nm_embs
        results["nm_latency_mean"] = float(np.mean(latencies))
        results["nm_latency_median"] = float(np.median(latencies))
        results["nm_latency_p90"] = float(np.percentile(latencies, 90))
        results["nm_peak_mem_mb"] = nm_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Normal FAILED: {e}")
        traceback.print_exc()
        results["nm_ok"] = False

    # Compare embeddings
    compare_embedding_results(results)
    return results


def bench_text_rerank(name, cfg, gpu_id, num_warmup=3, num_iter=20):
    """Benchmark text reranker model."""
    model_path = cfg["path"]
    if not os.path.exists(model_path):
        return None

    results = {"name": name, "category": cfg["category"], "model_path": model_path}
    pairs = [
        ("What is the capital of China?", "The capital of China is Beijing."),
        ("Explain gravity", "Gravity is a force that attracts two bodies."),
        ("What is machine learning?", "Machine learning is a subset of artificial intelligence."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
        ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
    ]

    # 1. HF baseline
    print(f"  [1/3] HuggingFace Transformers ...")
    reset_memory_stats()
    try:
        ref_scores, hf_latencies, hf_peak = hf_text_rerank(model_path, pairs, num_warmup, num_iter)
        results["hf_ok"] = True
        results["hf_latency_mean"] = float(np.mean(hf_latencies))
        results["hf_latency_median"] = float(np.median(hf_latencies))
        results["hf_latency_p90"] = float(np.percentile(hf_latencies, 90))
        results["hf_peak_mem_mb"] = hf_peak
        results["ref_scores"] = ref_scores
    except Exception as e:
        print(f"  HF FAILED: {e}")
        traceback.print_exc()
        results["hf_ok"] = False
        ref_scores = None

    # 2. Prefill-Only
    print(f"  [2/3] nano-vllm Prefill-Only ...")
    reset_memory_stats()
    try:
        from nanovllm import LLM
        llm_kwargs = dict(cfg["llm_kwargs"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm = LLM(model_path, **llm_kwargs)

        # Warmup
        for _ in range(num_warmup):
            llm.rerank_batch(pairs[:1], use_tqdm=False)

        po_scores, _ = get_nanovllm_scores(llm, pairs)

        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.rerank_batch(pairs, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        po_peak = get_peak_mb()
        results["po_ok"] = True
        results["po_scores"] = po_scores
        results["po_latency_mean"] = float(np.mean(latencies))
        results["po_latency_median"] = float(np.median(latencies))
        results["po_latency_p90"] = float(np.percentile(latencies, 90))
        results["po_peak_mem_mb"] = po_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Prefill-Only FAILED: {e}")
        traceback.print_exc()
        results["po_ok"] = False

    # 3. Normal
    print(f"  [3/3] nano-vllm Normal ...")
    reset_memory_stats()
    try:
        from nanovllm import LLM
        llm_kwargs = dict(cfg["llm_kwargs_normal"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm = LLM(model_path, **llm_kwargs)

        nm_scores, _ = get_nanovllm_scores(llm, pairs)

        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.rerank_batch(pairs, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        nm_peak = get_peak_mb()
        results["nm_ok"] = True
        results["nm_scores"] = nm_scores
        results["nm_latency_mean"] = float(np.mean(latencies))
        results["nm_latency_median"] = float(np.median(latencies))
        results["nm_latency_p90"] = float(np.percentile(latencies, 90))
        results["nm_peak_mem_mb"] = nm_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Normal FAILED: {e}")
        traceback.print_exc()
        results["nm_ok"] = False

    compare_reranker_results(results)
    return results


def bench_vl_embed(name, cfg, gpu_id, num_warmup=2, num_iter=5):
    """Benchmark VL embedding model (prefill-only vs normal)."""
    model_path = cfg["path"]
    if not os.path.exists(model_path):
        return None

    results = {"name": name, "category": cfg["category"], "model_path": model_path}
    texts = ["A bear in the image", "An outdoor scene", "Multiple bears"]
    image_urls = (
        "http://images.cocodataset.org/val2017/000000000285.jpg",
        "http://images.cocodataset.org/val2017/000000000632.jpg",
        "http://images.cocodataset.org/val2017/000000000724.jpg",
    )

    print("  Downloading test images ...")
    images = [download_image(url) for url in image_urls[:len(texts)]]

    # 2. Prefill-Only
    print(f"  [1/2] nano-vllm Prefill-Only ...")
    reset_memory_stats()
    try:
        from nanovllm import LLM
        llm_kwargs = dict(cfg["llm_kwargs"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm = LLM(model_path, **llm_kwargs)

        for _ in range(num_warmup):
            llm.embed_batch(texts[:1], images=images[:1], use_tqdm=False)

        po_embs, _ = get_nanovllm_embeddings(llm, texts, is_multimodal=True, images=images)

        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.embed_batch(texts, images=images, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        po_peak = get_peak_mb()
        results["po_ok"] = True
        results["po_embs"] = po_embs
        results["po_latency_mean"] = float(np.mean(latencies))
        results["po_latency_median"] = float(np.median(latencies))
        results["po_latency_p90"] = float(np.percentile(latencies, 90))
        results["po_peak_mem_mb"] = po_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Prefill-Only FAILED: {e}")
        traceback.print_exc()
        results["po_ok"] = False

    # 3. Normal
    print(f"  [2/2] nano-vllm Normal ...")
    reset_memory_stats()
    try:
        from nanovllm import LLM
        llm_kwargs = dict(cfg["llm_kwargs_normal"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm = LLM(model_path, **llm_kwargs)

        nm_embs, _ = get_nanovllm_embeddings(llm, texts, is_multimodal=True, images=images)

        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.embed_batch(texts, images=images, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        nm_peak = get_peak_mb()
        results["nm_ok"] = True
        results["nm_embs"] = nm_embs
        results["nm_latency_mean"] = float(np.mean(latencies))
        results["nm_latency_median"] = float(np.median(latencies))
        results["nm_latency_p90"] = float(np.percentile(latencies, 90))
        results["nm_peak_mem_mb"] = nm_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Normal FAILED: {e}")
        traceback.print_exc()
        results["nm_ok"] = False

    compare_embedding_results(results)
    return results


def bench_vl_rerank(name, cfg, gpu_id, num_warmup=2, num_iter=5):
    """Benchmark VL reranker model (prefill-only vs normal)."""
    model_path = cfg["path"]
    if not os.path.exists(model_path):
        return None

    results = {"name": name, "category": cfg["category"], "model_path": model_path}
    pairs = [
        ("A bear in the image", "A bear standing in nature"),
        ("An outdoor scene", "The scene is taken outdoors"),
        ("Multiple bears", "Several bears in the wild"),
    ]
    image_urls = (
        "http://images.cocodataset.org/val2017/000000000285.jpg",
        "http://images.cocodataset.org/val2017/000000000632.jpg",
        "http://images.cocodataset.org/val2017/000000000724.jpg",
    )

    print("  Downloading test images ...")
    images = [download_image(url) for url in image_urls[:len(pairs)]]

    # 2. Prefill-Only
    print(f"  [1/2] nano-vllm Prefill-Only ...")
    reset_memory_stats()
    try:
        from nanovllm import LLM
        llm_kwargs = dict(cfg["llm_kwargs"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm = LLM(model_path, **llm_kwargs)

        for _ in range(num_warmup):
            llm.rerank_batch(pairs[:1], images=images[:1], use_tqdm=False)

        po_scores, _ = get_nanovllm_scores(llm, pairs, is_multimodal=True, images=images)

        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.rerank_batch(pairs, images=images, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        po_peak = get_peak_mb()
        results["po_ok"] = True
        results["po_scores"] = po_scores
        results["po_latency_mean"] = float(np.mean(latencies))
        results["po_latency_median"] = float(np.median(latencies))
        results["po_latency_p90"] = float(np.percentile(latencies, 90))
        results["po_peak_mem_mb"] = po_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Prefill-Only FAILED: {e}")
        traceback.print_exc()
        results["po_ok"] = False

    # 3. Normal
    print(f"  [2/2] nano-vllm Normal ...")
    reset_memory_stats()
    try:
        from nanovllm import LLM
        llm_kwargs = dict(cfg["llm_kwargs_normal"])
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["tensor_parallel_size"] = 1
        llm = LLM(model_path, **llm_kwargs)

        nm_scores, _ = get_nanovllm_scores(llm, pairs, is_multimodal=True, images=images)

        latencies = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.rerank_batch(pairs, images=images, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        nm_peak = get_peak_mb()
        results["nm_ok"] = True
        results["nm_scores"] = nm_scores
        results["nm_latency_mean"] = float(np.mean(latencies))
        results["nm_latency_median"] = float(np.median(latencies))
        results["nm_latency_p90"] = float(np.percentile(latencies, 90))
        results["nm_peak_mem_mb"] = nm_peak

        cleanup_llm(llm)
    except Exception as e:
        print(f"  Normal FAILED: {e}")
        traceback.print_exc()
        results["nm_ok"] = False

    compare_reranker_results(results)
    return results


# ── Comparison helpers ─────────────────────────────────────────────────────────

def compare_logits_results(results, ref_logits, po_logits, nm_logits):
    """Compare logits between modes and store metrics in results dict."""
    # Prefill-Only vs HF
    if ref_logits is not None and po_logits is not None:
        try:
            # Ensure same shape
            min_vocab = min(ref_logits.shape[-1], po_logits.shape[-1])
            rl = ref_logits[:, :min_vocab]
            pl = po_logits[:, :min_vocab] if po_logits.dim() == 2 else po_logits

            if rl.shape == pl.shape:
                cos_sim = torch.nn.functional.cosine_similarity(rl, pl, dim=-1)
                results["po_vs_hf_cos_sim_mean"] = float(cos_sim.mean())
                results["po_vs_hf_cos_sim_min"] = float(cos_sim.min())
                results["po_vs_hf_max_diff"] = float((rl - pl).abs().max())
                results["po_vs_hf_mean_diff"] = float((rl - pl).abs().mean())
                # Token match
                po_argmax = pl.argmax(dim=-1).tolist()
                hf_argmax = rl.argmax(dim=-1).tolist()
                matches = sum(1 for a, b in zip(po_argmax, hf_argmax) if a == b)
                results["po_vs_hf_token_match"] = f"{matches}/{len(hf_argmax)}"
            else:
                results["po_vs_hf_error"] = f"Shape mismatch: {rl.shape} vs {pl.shape}"
        except Exception as e:
            results["po_vs_hf_error"] = str(e)

    # Normal vs HF
    if ref_logits is not None and nm_logits is not None:
        try:
            min_vocab = min(ref_logits.shape[-1], nm_logits.shape[-1])
            rl = ref_logits[:, :min_vocab]
            nl = nm_logits[:, :min_vocab] if nm_logits.dim() == 2 else nm_logits

            if rl.shape == nl.shape:
                cos_sim = torch.nn.functional.cosine_similarity(rl, nl, dim=-1)
                results["nm_vs_hf_cos_sim_mean"] = float(cos_sim.mean())
                results["nm_vs_hf_cos_sim_min"] = float(cos_sim.min())
                results["nm_vs_hf_max_diff"] = float((rl - nl).abs().max())
                results["nm_vs_hf_mean_diff"] = float((rl - nl).abs().mean())
                nm_argmax = nl.argmax(dim=-1).tolist()
                hf_argmax = rl.argmax(dim=-1).tolist()
                matches = sum(1 for a, b in zip(nm_argmax, hf_argmax) if a == b)
                results["nm_vs_hf_token_match"] = f"{matches}/{len(hf_argmax)}"
            else:
                results["nm_vs_hf_error"] = f"Shape mismatch: {rl.shape} vs {nl.shape}"
        except Exception as e:
            results["nm_vs_hf_error"] = str(e)

    # Prefill-Only vs Normal (most important!)
    if po_logits is not None and nm_logits is not None:
        try:
            min_vocab = min(po_logits.shape[-1], nm_logits.shape[-1])
            # Handle dimension mismatch
            if po_logits.dim() == 1:
                po_logits_2d = po_logits.unsqueeze(0)[:, :min_vocab]
            else:
                po_logits_2d = po_logits[:, :min_vocab]
            if nm_logits.dim() == 1:
                nm_logits_2d = nm_logits.unsqueeze(0)[:, :min_vocab]
            else:
                nm_logits_2d = nm_logits[:, :min_vocab]

            # Match batch size
            min_batch = min(po_logits_2d.shape[0], nm_logits_2d.shape[0])
            pl = po_logits_2d[:min_batch]
            nl = nm_logits_2d[:min_batch]

            if pl.shape == nl.shape:
                cos_sim = torch.nn.functional.cosine_similarity(pl, nl, dim=-1)
                results["po_vs_nm_cos_sim_mean"] = float(cos_sim.mean())
                results["po_vs_nm_cos_sim_min"] = float(cos_sim.min())
                results["po_vs_nm_max_diff"] = float((pl - nl).abs().max())
                results["po_vs_nm_mean_diff"] = float((pl - nl).abs().mean())
                po_argmax = pl.argmax(dim=-1).tolist()
                nm_argmax = nl.argmax(dim=-1).tolist()
                matches = sum(1 for a, b in zip(po_argmax, nm_argmax) if a == b)
                results["po_vs_nm_token_match"] = f"{matches}/{len(nm_argmax)}"
            else:
                results["po_vs_nm_error"] = f"Shape mismatch: {pl.shape} vs {nl.shape}"
        except Exception as e:
            results["po_vs_nm_error"] = str(e)


def compare_embedding_results(results):
    """Compare embeddings between prefill-only and normal modes."""
    po_embs = results.get("po_embs")
    nm_embs = results.get("nm_embs")
    ref_embs = results.get("ref_embs")

    if ref_embs is not None and po_embs is not None:
        try:
            min_dim = min(ref_embs.shape[-1], po_embs.shape[-1])
            min_batch = min(ref_embs.shape[0], po_embs.shape[0])
            re = ref_embs[:min_batch, :min_dim]
            pe = po_embs[:min_batch, :min_dim]
            re = torch.nn.functional.normalize(re.float(), p=2, dim=-1)
            pe = torch.nn.functional.normalize(pe.float(), p=2, dim=-1)
            cos_sim = torch.nn.functional.cosine_similarity(re, pe, dim=-1)
            results["po_vs_hf_cos_sim_mean"] = float(cos_sim.mean())
            results["po_vs_hf_cos_sim_min"] = float(cos_sim.min())
            results["po_vs_hf_max_diff"] = float((re - pe).abs().max())
        except Exception as e:
            results["po_vs_hf_emb_error"] = str(e)

    if po_embs is not None and nm_embs is not None:
        try:
            min_dim = min(po_embs.shape[-1], nm_embs.shape[-1])
            min_batch = min(po_embs.shape[0], nm_embs.shape[0])
            pe = po_embs[:min_batch, :min_dim].float()
            ne = nm_embs[:min_batch, :min_dim].float()
            pe = torch.nn.functional.normalize(pe, p=2, dim=-1)
            ne = torch.nn.functional.normalize(ne, p=2, dim=-1)
            cos_sim = torch.nn.functional.cosine_similarity(pe, ne, dim=-1)
            results["po_vs_nm_cos_sim_mean"] = float(cos_sim.mean())
            results["po_vs_nm_cos_sim_min"] = float(cos_sim.min())
            results["po_vs_nm_max_diff"] = float((pe - ne).abs().max())
        except Exception as e:
            results["po_vs_nm_emb_error"] = str(e)


def compare_reranker_results(results):
    """Compare reranker scores between prefill-only and normal modes."""
    po_scores = results.get("po_scores")
    nm_scores = results.get("nm_scores")
    ref_scores = results.get("ref_scores")

    if ref_scores is not None and po_scores is not None:
        try:
            min_batch = min(ref_scores.shape[0], po_scores.shape[0])
            rs = ref_scores[:min_batch].float()
            ps = po_scores[:min_batch].float()
            results["po_vs_hf_max_diff"] = float((rs - ps).abs().max())
            results["po_vs_hf_mean_diff"] = float((rs - ps).abs().mean())
            rel_err = (rs - ps).abs() / (rs.abs() + 1e-8)
            results["po_vs_hf_mean_rel_err"] = float(rel_err.mean())
        except Exception as e:
            results["po_vs_hf_score_error"] = str(e)

    if po_scores is not None and nm_scores is not None:
        try:
            min_batch = min(po_scores.shape[0], nm_scores.shape[0])
            ps = po_scores[:min_batch].float()
            ns = nm_scores[:min_batch].float()
            results["po_vs_nm_max_diff"] = float((ps - ns).abs().max())
            results["po_vs_nm_mean_diff"] = float((ps - ns).abs().mean())
            rel_err = (ps - ns).abs() / (ps.abs() + 1e-8)
            results["po_vs_nm_mean_rel_err"] = float(rel_err.mean())
        except Exception as e:
            results["po_vs_nm_score_error"] = str(e)


# ── Report generation ──────────────────────────────────────────────────────────

def generate_report(all_results, output_path):
    """Generate markdown report from benchmark results."""
    lines = []
    lines.append("# nano-vllm-prefillonly Benchmark Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Overview\n")
    lines.append("Compares three inference modes per model:")
    lines.append("1. **HuggingFace Transformers** (ground truth)")
    lines.append("2. **nano-vllm Prefill-Only** (optimized, no KV cache)")
    lines.append("3. **nano-vllm Normal** (with KV cache)\n")
    lines.append("Metrics: Logits/embeddings/scores consistency, Speed, VRAM\n")

    # Summary table
    lines.append("## Summary Table\n")
    lines.append("| Model | Category | Prefill vs Normal Consistency | Speed (PO/NM) | VRAM PO | VRAM NM | VRAM Saved |")
    lines.append("|-------|----------|-------------------------------|----------------|---------|---------|------------|")

    for r in all_results:
        if r is None:
            continue
        name = r["name"]
        cat = r["category"]

        # Consistency
        consistency = "N/A"
        if "po_vs_nm_cos_sim_mean" in r:
            consistency = f"cos={r['po_vs_nm_cos_sim_mean']:.4f}"
        elif "po_vs_nm_max_diff" in r:
            consistency = f"max_diff={r['po_vs_nm_max_diff']:.6f}"
        elif "po_vs_nm_token_match" in r:
            consistency = f"match={r['po_vs_nm_token_match']}"

        # Speed
        po_speed = f"{r.get('po_latency_mean', 0)*1000:.1f}ms" if r.get("po_ok") else "FAIL"
        nm_speed = f"{r.get('nm_latency_mean', 0)*1000:.1f}ms" if r.get("nm_ok") else "FAIL"
        speed = f"{po_speed} / {nm_speed}"

        # VRAM
        po_mem = f"{r.get('po_peak_mem_mb', 0):.0f}MB" if r.get("po_ok") else "N/A"
        nm_mem = f"{r.get('nm_peak_mem_mb', 0):.0f}MB" if r.get("nm_ok") else "N/A"

        saved = "N/A"
        if r.get("po_ok") and r.get("nm_ok"):
            po_m = r.get("po_peak_mem_mb", 0)
            nm_m = r.get("nm_peak_mem_mb", 0)
            if nm_m > 0:
                saved = f"{(1 - po_m/nm_m)*100:.1f}%"

        lines.append(f"| {name} | {cat} | {consistency} | {speed} | {po_mem} | {nm_mem} | {saved} |")

    lines.append("")

    # Detailed results per model
    for r in all_results:
        if r is None:
            continue
        lines.append(f"## {r['name']}\n")
        lines.append(f"- **Category**: {r['category']}")
        lines.append(f"- **Model Path**: `{r['model_path']}`\n")

        # 1. Consistency
        lines.append("### 1. Output Consistency\n")

        if r["category"] in ("text_gen", "vl_gen"):
            if "po_vs_hf_cos_sim_mean" in r:
                lines.append(f"**Prefill-Only vs HuggingFace:**")
                lines.append(f"- Cosine Similarity: mean={r['po_vs_hf_cos_sim_mean']:.6f}, min={r['po_vs_hf_cos_sim_min']:.6f}")
                lines.append(f"- Max Diff: {r['po_vs_hf_max_diff']:.6f}")
                lines.append(f"- Mean Diff: {r['po_vs_hf_mean_diff']:.6f}")
                if "po_vs_hf_token_match" in r:
                    lines.append(f"- Token Match: {r['po_vs_hf_token_match']}")
                lines.append("")

            if "nm_vs_hf_cos_sim_mean" in r:
                lines.append(f"**Normal vs HuggingFace:**")
                lines.append(f"- Cosine Similarity: mean={r['nm_vs_hf_cos_sim_mean']:.6f}, min={r['nm_vs_hf_cos_sim_min']:.6f}")
                lines.append(f"- Max Diff: {r['nm_vs_hf_max_diff']:.6f}")
                if "nm_vs_hf_token_match" in r:
                    lines.append(f"- Token Match: {r['nm_vs_hf_token_match']}")
                lines.append("")

            if "po_vs_nm_cos_sim_mean" in r:
                lines.append(f"**Prefill-Only vs Normal (Key Comparison):**")
                lines.append(f"- Cosine Similarity: mean={r['po_vs_nm_cos_sim_mean']:.6f}, min={r['po_vs_nm_cos_sim_min']:.6f}")
                lines.append(f"- Max Diff: {r['po_vs_nm_max_diff']:.6f}")
                lines.append(f"- Mean Diff: {r['po_vs_nm_mean_diff']:.6f}")
                if "po_vs_nm_token_match" in r:
                    lines.append(f"- Token Match: {r['po_vs_nm_token_match']}")
                lines.append("")

            for key in ("po_vs_hf_error", "nm_vs_hf_error", "po_vs_nm_error"):
                if key in r:
                    lines.append(f"- {key}: {r[key]}")

        elif r["category"] in ("text_embed", "vl_embed"):
            if "po_vs_hf_cos_sim_mean" in r:
                lines.append(f"**Prefill-Only vs HuggingFace:**")
                lines.append(f"- Cosine Similarity: mean={r['po_vs_hf_cos_sim_mean']:.6f}, min={r['po_vs_hf_cos_sim_min']:.6f}")
                lines.append(f"- Max Diff: {r['po_vs_hf_max_diff']:.6f}")
                lines.append("")
            if "po_vs_nm_cos_sim_mean" in r:
                lines.append(f"**Prefill-Only vs Normal:**")
                lines.append(f"- Cosine Similarity: mean={r['po_vs_nm_cos_sim_mean']:.6f}, min={r['po_vs_nm_cos_sim_min']:.6f}")
                lines.append(f"- Max Diff: {r['po_vs_nm_max_diff']:.6f}")
                lines.append("")

        elif r["category"] in ("text_rerank", "vl_rerank"):
            if "po_vs_hf_max_diff" in r:
                lines.append(f"**Prefill-Only vs HuggingFace:**")
                lines.append(f"- Max Score Diff: {r['po_vs_hf_max_diff']:.6f}")
                lines.append(f"- Mean Score Diff: {r['po_vs_hf_mean_diff']:.6f}")
                if "po_vs_hf_mean_rel_err" in r:
                    lines.append(f"- Mean Relative Error: {r['po_vs_hf_mean_rel_err']:.6f}")
                lines.append("")
            if "po_vs_nm_max_diff" in r:
                lines.append(f"**Prefill-Only vs Normal:**")
                lines.append(f"- Max Score Diff: {r['po_vs_nm_max_diff']:.6f}")
                lines.append(f"- Mean Score Diff: {r['po_vs_nm_mean_diff']:.6f}")
                if "po_vs_nm_mean_rel_err" in r:
                    lines.append(f"- Mean Relative Error: {r['po_vs_nm_mean_rel_err']:.6f}")
                lines.append("")

        # 2. Speed
        lines.append("### 2. Speed\n")
        lines.append("| Mode | Mean (ms) | Median (ms) | P90 (ms) |")
        lines.append("|------|-----------|-------------|----------|")
        if r.get("hf_ok"):
            lines.append(f"| HF Transformers | {r['hf_latency_mean']*1000:.1f} | {r['hf_latency_median']*1000:.1f} | {r['hf_latency_p90']*1000:.1f} |")
        if r.get("po_ok"):
            lines.append(f"| Prefill-Only | {r['po_latency_mean']*1000:.1f} | {r['po_latency_median']*1000:.1f} | {r['po_latency_p90']*1000:.1f} |")
        if r.get("nm_ok"):
            lines.append(f"| Normal | {r['nm_latency_mean']*1000:.1f} | {r['nm_latency_median']*1000:.1f} | {r['nm_latency_p90']*1000:.1f} |")
        lines.append("")

        # 3. VRAM
        lines.append("### 3. VRAM (Peak)\n")
        lines.append("| Mode | Peak Memory (MB) |")
        lines.append("|------|------------------|")
        if r.get("hf_ok"):
            lines.append(f"| HF Transformers | {r['hf_peak_mem_mb']:.0f} |")
        if r.get("po_ok"):
            lines.append(f"| Prefill-Only | {r['po_peak_mem_mb']:.0f} |")
        if r.get("nm_ok"):
            lines.append(f"| Normal | {r['nm_peak_mem_mb']:.0f} |")
        if r.get("po_ok") and r.get("nm_ok") and r["nm_peak_mem_mb"] > 0:
            ratio = r["nm_peak_mem_mb"] / r["po_peak_mem_mb"]
            pct = (1 - r["po_peak_mem_mb"] / r["nm_peak_mem_mb"]) * 100
            lines.append(f"\n**VRAM Savings**: Prefill-Only uses **{ratio:.2f}x** less memory ({pct:.1f}% saved)")
        lines.append("\n---\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to: {output_path}")


# ── Runner that uses subprocess isolation ──────────────────────────────────────

def run_model_in_subprocess(model_name, cfg, gpu_id, result_file):
    """Run a single model benchmark in an isolated subprocess.

    This prevents GPU memory leaks between models.
    """
    script = f'''
import sys
sys.path.insert(0, "{os.getcwd()}")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

import json
import torch

# Re-import everything in subprocess
from benchmark_all import MODEL_CONFIGS, bench_text_gen, bench_vl_gen, bench_text_embed, bench_text_rerank, bench_vl_embed, bench_vl_rerank

name = "{model_name}"
cfg = MODEL_CONFIGS[name]
cat = cfg["category"]

print(f"\\n{'='*60}")
print(f"Benchmarking: {{name}} ({{cat}})")
print(f"{'='*60}")

result = None
try:
    if cat == "text_gen":
        result = bench_text_gen(name, cfg, {gpu_id})
    elif cat == "vl_gen":
        result = bench_vl_gen(name, cfg, {gpu_id})
    elif cat == "text_embed":
        result = bench_text_embed(name, cfg, {gpu_id})
    elif cat == "vl_embed":
        result = bench_vl_embed(name, cfg, {gpu_id})
    elif cat == "text_rerank":
        result = bench_text_rerank(name, cfg, {gpu_id})
    elif cat == "vl_rerank":
        result = bench_vl_rerank(name, cfg, {gpu_id})
except Exception as e:
    print(f"FAILED: {{e}}")
    import traceback
    traceback.print_exc()

# Save result (strip non-serializable tensors)
if result is not None:
    serializable = {{}}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            serializable[k] = v.tolist()
        else:
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)
    with open("{result_file}", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Result saved to {result_file}")
else:
    with open("{result_file}", "w") as f:
        json.dump({{"name": name, "error": "benchmark failed"}}, f, indent=2)
'''
    return script


def main():
    parser = argparse.ArgumentParser(description="Benchmark all models")
    parser.add_argument("--category", type=str, default="all",
                        choices=CATEGORIES + ["all"],
                        help="Category to benchmark")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to benchmark")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip models not found locally")
    parser.add_argument("--output", type=str, default="BENCHMARK_RESULTS.md",
                        help="Output report file")
    parser.add_argument("--in-process", action="store_true",
                        help="Run in-process (no subprocess isolation)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Filter models
    models_to_run = []
    for name, cfg in MODEL_CONFIGS.items():
        if args.category != "all" and cfg["category"] != args.category:
            continue
        if args.model and name != args.model:
            continue
        if args.skip_missing and not os.path.exists(cfg["path"]):
            print(f"SKIP (not found): {name} @ {cfg['path']}")
            continue
        models_to_run.append((name, cfg))

    print(f"Models to benchmark: {len(models_to_run)}")
    for name, _ in models_to_run:
        print(f"  - {name}")

    all_results = []

    if args.in_process:
        # Run directly in this process
        for name, cfg in models_to_run:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {name} ({cfg['category']})")
            print(f"{'='*60}")

            result = None
            cat = cfg["category"]
            try:
                if cat == "text_gen":
                    result = bench_text_gen(name, cfg, args.gpu)
                elif cat == "vl_gen":
                    result = bench_vl_gen(name, cfg, args.gpu)
                elif cat == "text_embed":
                    result = bench_text_embed(name, cfg, args.gpu)
                elif cat == "vl_embed":
                    result = bench_vl_embed(name, cfg, args.gpu)
                elif cat == "text_rerank":
                    result = bench_text_rerank(name, cfg, args.gpu)
                elif cat == "vl_rerank":
                    result = bench_vl_rerank(name, cfg, args.gpu)
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()

            all_results.append(result)
    else:
        # Run each model in isolated subprocess
        import tempfile
        for name, cfg in models_to_run:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {name} ({cfg['category']})")
            print(f"{'='*60}")

            result_file = os.path.join(tempfile.gettempdir(), f"bench_{name}.json")
            script = run_model_in_subprocess(name, cfg, args.gpu, result_file)

            script_file = os.path.join(tempfile.gettempdir(), f"bench_{name}.py")
            with open(script_file, "w") as f:
                f.write(script)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

            try:
                result = subprocess.run(
                    [sys.executable, script_file],
                    env=env, capture_output=False, timeout=600,
                )

                if os.path.exists(result_file):
                    with open(result_file) as f:
                        all_results.append(json.load(f))
                else:
                    all_results.append({"name": name, "error": "no result file"})
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT: {name}")
                all_results.append({"name": name, "error": "timeout"})
            except Exception as e:
                print(f"SUBPROCESS ERROR: {e}")
                all_results.append({"name": name, "error": str(e)})

    # Generate report
    generate_report(all_results, args.output)


if __name__ == "__main__":
    main()
