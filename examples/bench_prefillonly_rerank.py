"""Comprehensive test script for prefill-only reranking optimizations.

This script provides comprehensive comparisons for reranking tasks:
1. Accuracy comparison with transformers baseline
2. Speed comparison with transformers baseline
3. Memory usage comparison with transformers baseline
4. Speed comparison: prefill-only vs original nano-vllm
5. Memory comparison: prefill-only vs original nano-vllm

Models tested:
- Text-only: Qwen3-Reranker-0.6B
- Multimodal: jina-reranker-m0

Usage:
    python bench_prefillonly_rerank.py --modality text
    python bench_prefillonly_rerank.py --modality multimodal
    python bench_prefillonly_rerank.py --modality all --model jina-reranker-m0
    python bench_prefillonly_rerank.py --batch-size 32 --num-warmup 10
"""
import argparse
import io
import os
import time
import numpy as np
import torch
import torch.distributed as dist
import urllib.request
from typing import Callable, Dict, List, Optional
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM,
)

from nanovllm import LLM
from examples.prompt import (
    DEFAULT_IMAGE_URLS,
    MULTIMODAL_RERANKING_QUERIES,
    TEXT_RERANKING_QUERIES,
    TEXT_RERANKING_DOCUMENTS,
)


# Global variable to store model filter
_MODEL_FILTER = None


def get_model_dtype(model_path: str) -> torch.dtype:
    """Get model dtype from config, matching nano-vllm's logic.
    
    Args:
        model_path: Path to the model
        
    Returns:
        torch.dtype: The dtype to use for the model
    """
    from transformers import AutoConfig
    
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    torch_dtype = getattr(hf_config, "torch_dtype", None)
    if torch_dtype is None and hasattr(hf_config, "text_config"):
        torch_dtype = getattr(hf_config.text_config, "torch_dtype", None)
    if isinstance(torch_dtype, str):
        resolved_dtype = getattr(torch, torch_dtype, None)
        if resolved_dtype is None:
            alias_map = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
            }
            resolved_dtype = alias_map.get(torch_dtype.lower())
        torch_dtype = resolved_dtype
    if torch_dtype is None:
        torch_dtype = torch.float16
    return torch_dtype


def set_model_filter(model_name: Optional[str]):
    """Set global model filter."""
    global _MODEL_FILTER
    _MODEL_FILTER = model_name


def should_test_model(model_name: str) -> bool:
    """Check if model should be tested based on global filter."""
    if _MODEL_FILTER is None:
        return True
    model_lower = model_name.lower()
    filter_lower = _MODEL_FILTER.lower()
    
    # Model name mappings for flexible matching
    model_name_mappings = {
        "jina-reranker-m0": ["jina-reranker-m0", "jina_reranker_m0"],
        "jina-embedding-v4": [
            "jina_embedding_v4",
            "jina-embeddings-v4",
            "jina_embeddings_v4",
        ],
        "qwen2vl": ["qwen2vl", "qwen2_vl", "qwen2-vl"],
        "qwen2.5vl": ["qwen2.5vl", "qwen2_5_vl", "qwen2.5-vl"],
        "qwen3vl": ["qwen3vl", "qwen3_vl", "qwen3-vl"],
        "llavanext": ["llavanext", "llava-v1.6", "llava_v1.6"],
    }
    
    if filter_lower in model_name_mappings:
        return any(
            m in model_lower for m in model_name_mappings[filter_lower]
        )
    return filter_lower in model_lower


def get_reranker_type_from_model(model_name: str) -> tuple[str, bool]:
    """Get reranker type and is_listwise flag from model name.
    
    Args:
        model_name: Model name (e.g., 'qwen3-reranker', 'jina-reranker-v3', 'bge-reranker-v2-gemma')
        
    Returns:
        (reranker_type, is_listwise): Tuple of reranker type and listwise flag
    """
    model_lower = model_name.lower()
    
    # Model name to reranker type mapping
    if "jina-reranker-v3" in model_lower or "jina_reranker_v3" in model_lower:
        return ("jina_v3", True)  # jina_v3 is listwise
    elif "bge-reranker-v2-gemma" in model_lower or "bge_reranker_v2_gemma" in model_lower:
        return ("gemma", False)  # gemma is pointwise
    elif "qwen3-reranker" in model_lower or "qwen3_reranker" in model_lower or "jina-reranker-m0" in model_lower:
        return ("qwen3", False)  # qwen3 and jina-m0 are pointwise
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: qwen3-reranker, jina-reranker-v3, bge-reranker-v2-gemma, jina-reranker-m0"
        )


def sanitize_input(text: str, special_tokens: dict) -> str:
    """Remove special tokens from text to avoid duplication."""
    for token in special_tokens.values():
        text = text.replace(token, "")
    return text


def format_docs_prompts_func(
    query: str,
    docs: List[str],
    instruction: Optional[str] = None,
    special_tokens: Optional[dict] = None,
    no_thinking: bool = True,
) -> str:
    """Format input for Jina-Reranker-V3 (listwise reranker)."""
    if special_tokens is None:
        special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>",
        }

    # Sanitize inputs to remove any existing special tokens
    query = sanitize_input(query, special_tokens)
    docs = [sanitize_input(doc, special_tokens) for doc in docs]

    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of "
        "the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on "
        "how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well "
        "each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction "
        "when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    if no_thinking:
        suffix += "<think>\n\n</think>\n\n"

    doc_emb_token = special_tokens["doc_embed_token"]
    query_emb_token = special_tokens["query_embed_token"]

    prompt = (
        f"I will provide you with {len(docs)} passages, each indicated by "
        f"a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )

    if instruction:
        prompt += f"<instruct>\n{instruction}\n</instruct>\n"

    doc_prompts = [
        f'<passage id="{i}">\n{doc}{doc_emb_token}\n</passage>'
        for i, doc in enumerate(docs)
    ]
    prompt += "\n".join(doc_prompts) + "\n"
    prompt += f"<query>\n{query}{query_emb_token}\n</query>"

    return prefix + prompt + suffix


def get_gemma_inputs(
    pairs: List[tuple],
    tokenizer,
    prompt: Optional[str] = None,
    max_length: int = 1024,
):
    """Format inputs for Gemma reranker (pointwise reranker)."""
    if prompt is None:
        prompt = (
            "Given a query A and a passage B, determine whether the passage "
            "contains an answer to the query by providing a prediction of "
            "either 'Yes' or 'No'."
        )

    sep = "\n"
    prompt_inputs = tokenizer(
        prompt, return_tensors=None, add_special_tokens=False
    )['input_ids']
    sep_inputs = tokenizer(
        sep, return_tensors=None, add_special_tokens=False
    )['input_ids']

    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(
            f'A: {query}',
            return_tensors=None,
            add_special_tokens=False,
            max_length=max_length * 3 // 4,
            truncation=True
        )
        passage_inputs = tokenizer(
            f'B: {passage}',
            return_tensors=None,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True
        )
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)

    return tokenizer.pad(
        inputs,
        padding=True,
        max_length=max_length + len(sep_inputs) + len(prompt_inputs),
        pad_to_multiple_of=8,
        return_tensors='pt',
    )


def download_image(url: str) -> Image.Image:
    """Download image from URL."""
    with urllib.request.urlopen(url) as response:
        data = response.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_peak_memory_usage():
    """Get peak GPU memory usage in MB (since last reset_peak_memory_stats).

    Note: This returns the peak since the last call to
    torch.cuda.reset_peak_memory_stats(), not the global peak.
    Uses memory_stats()["allocated_bytes.all.peak"] which resets correctly.
    """
    if torch.cuda.is_available():
        stats = torch.cuda.memory_stats()
        # Use allocated_bytes.all.peak which resets with
        # reset_peak_memory_stats()
        peak_bytes = stats.get("allocated_bytes.all.peak", 0)
        if peak_bytes == 0:
            # Fallback: if peak is 0, it might mean no allocation happened
            # or stats weren't reset. Use current allocated as fallback.
            peak_bytes = torch.cuda.memory_allocated()
        return peak_bytes / 1024 / 1024
    return 0


def compare_results(
    transformers_result: torch.Tensor | list,
    nanovllm_result: torch.Tensor | list,
    task_type: str = "embedding",
) -> Dict[str, float]:
    """Compare results between transformers and nano-vllm.

    Returns:
        dict with comparison metrics (cosine_sim, max_diff, mean_diff, etc.)
    """
    if task_type == "embedding":
        # For embeddings, compute cosine similarity
        if isinstance(transformers_result, list):
            hf_emb = torch.tensor(transformers_result).cpu().float()
        else:
            hf_emb = transformers_result.cpu().float()
        if isinstance(nanovllm_result, list):
            nv_emb = torch.tensor(nanovllm_result).cpu().float()
        else:
            nv_emb = nanovllm_result.cpu().float()

        # Normalize
        hf_emb = torch.nn.functional.normalize(hf_emb, p=2, dim=1)
        nv_emb = torch.nn.functional.normalize(nv_emb, p=2, dim=1)

        cosine_sim = (hf_emb * nv_emb).sum(dim=1)
        max_diff = (hf_emb - nv_emb).abs().max().item()
        mean_diff = (hf_emb - nv_emb).abs().mean().item()

        return {
            "cosine_similarity_mean": cosine_sim.mean().item(),
            "cosine_similarity_min": cosine_sim.min().item(),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        }
    elif task_type == "reranking":
        # For reranking, compare scores
        if isinstance(transformers_result, list):
            hf_scores = torch.tensor(transformers_result)
        else:
            hf_scores = transformers_result

        if isinstance(nanovllm_result, list):
            nv_scores = torch.tensor(nanovllm_result)
        else:
            nv_scores = nanovllm_result

        max_diff = (hf_scores - nv_scores).abs().max().item()
        mean_diff = (hf_scores - nv_scores).abs().mean().item()
        relative_error = (hf_scores - nv_scores).abs() / (
            hf_scores.abs() + 1e-8
        )

        return {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "max_relative_error": relative_error.max().item(),
            "mean_relative_error": relative_error.mean().item(),
        }
    elif task_type == "generation":
        # For generation, compare token IDs
        matches = sum(
            1
            for hf, nv in zip(transformers_result, nanovllm_result)
            if hf == nv
        )
        return {
            "exact_matches": matches,
            "total": len(transformers_result),
            "match_rate": matches / len(transformers_result)
            if transformers_result
            else 0,
        }
    return {}


def cleanup_distributed():
    """Clean up distributed process group if initialized."""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass  # Ignore errors during cleanup


def test_memory_comparison(model_path: Optional[str] = None):
    """Compare memory usage between prefill-only and normal mode."""
    print("\n" + "=" * 60)
    print("Memory Usage Comparison")
    print("=" * 60)

    if model_path is None:
        raise ValueError("--model-path is required for memory comparison test")
    model_path = os.path.expanduser(model_path)
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Skipping memory comparison test")
        return

    # Clear memory
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Test with prefill-only mode
    print("\n--- Prefill-Only Mode ---")
    llm_prefill = LLM(
        model_path,
        is_reranker=True,
        reranker_type="qwen3",
        is_original_qwen3_reranker=True,
        classifier_from_token=["no", "yes"],
        enforce_eager=True,
        tensor_parallel_size=1,
        prefill_only_mode=True,
    )

    query_doc_pairs = [
        ("What is the capital of China?", "The capital of China is Beijing."),
        ("Explain gravity", "Gravity is a force that attracts two bodies."),
    ] * 5
    _ = llm_prefill.rerank_batch(query_doc_pairs, use_tqdm=False)

    memory_prefill = get_memory_usage()
    peak_memory_prefill = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else 0
    )

    print(f"Memory usage: {memory_prefill:.2f} MB")
    print(f"Peak memory: {peak_memory_prefill:.2f} MB")

    llm_prefill.exit()
    cleanup_distributed()
    torch.cuda.empty_cache()

    # Test with normal mode
    print("\n--- Normal Mode (for comparison) ---")
    llm_original = LLM(
        model_path,
        is_reranker=True,
        reranker_type="qwen3",
        is_original_qwen3_reranker=True,
        classifier_from_token=["no", "yes"],
        enforce_eager=True,
        tensor_parallel_size=1,
        prefill_only_mode=False,
    )

    _ = llm_original.rerank_batch(query_doc_pairs, use_tqdm=False)

    memory_original = get_memory_usage()
    peak_memory_original = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else 0
    )

    print(f"Memory usage: {memory_original:.2f} MB")
    print(f"Peak memory: {peak_memory_original:.2f} MB")

    llm_original.exit()
    cleanup_distributed()
    torch.cuda.empty_cache()

    print(f"\nMemory saved by prefill-only: "
          f"{(peak_memory_original - peak_memory_prefill) / peak_memory_original * 100:.1f}%")


# ==================== Class-based Comprehensive Tests ====================


class ComprehensiveTestBase:
    """Base class for comprehensive tests."""

    def __init__(
        self,
        model_path: str,
        num_warmup: int = 5,
        num_iterations: int = 10,
    ):
        self.model_path = os.path.expanduser(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}"
            )
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations

    def setup_test(self):
        """Setup before each test run."""
        cleanup_distributed()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def teardown_test(self):
        """Cleanup after each test run."""
        cleanup_distributed()
        torch.cuda.empty_cache()

    def benchmark_with_warmup(
        self,
        func: Callable,
        description: str = "Benchmarking",
    ) -> Dict[str, float]:
        """Benchmark a function with warmup iterations (vLLM style).

        Args:
            func: Function to benchmark (should return result)
            description: Description for progress display

        Returns:
            Dict with timing statistics and the last result
        """
        # Warmup phase
        if self.num_warmup > 0:
            print(f"  Warming up ({self.num_warmup} iterations)...")
            for _ in range(self.num_warmup):
                _ = func()
            torch.cuda.synchronize()

        # Benchmark phase
        print(f"  Running benchmark ({self.num_iterations} iterations)...")
        latencies = []
        result = None
        for i in range(self.num_iterations):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            result = func()
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

        latencies = np.array(latencies)

        # Calculate statistics
        stats = {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p99": float(np.percentile(latencies, 99)),
        }

        # Store result for accuracy comparison
        stats["result"] = result

        return stats

    def test_transformers_baseline(self) -> Dict:
        """Test with transformers baseline.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def test_nanovllm_prefill_only(self) -> Dict:
        """Test with nano-vllm prefill-only mode."""
        raise NotImplementedError

    def test_nanovllm_original(self) -> Dict:
        """Test with nano-vllm original mode."""
        raise NotImplementedError

    def compare_results(self, results: Dict) -> None:
        """Compare and print results."""
        hf_result = results["transformers"]
        prefill_result = results["nanovllm_prefill"]
        original_result = results["nanovllm_original"]

        # Accuracy comparison
        print("\n📊 Accuracy Comparison:")
        acc_prefill = compare_results(
            hf_result.get("output", hf_result.get("result")),
            prefill_result.get("output", prefill_result.get("result")),
            task_type=self.task_type,
        )
        acc_original = compare_results(
            hf_result.get("output", hf_result.get("result")),
            original_result.get("output", original_result.get("result")),
            task_type=self.task_type,
        )

        self._print_accuracy_comparison(acc_prefill, acc_original)

        # Speed comparison
        print("\n⚡ Speed Comparison:")
        hf_time = hf_result.get("mean", hf_result.get("time", 0))
        prefill_time = prefill_result.get(
            "mean", prefill_result.get("time", 0)
        )
        original_time = original_result.get(
            "mean", original_result.get("time", 0)
        )

        print("  Transformers:")
        print(f"    Mean:     {hf_time:.4f}s")
        print(f"    Median:   {hf_result.get('median', 0):.4f}s")
        print(f"    P90:      {hf_result.get('p90', 0):.4f}s")
        print(f"    P99:      {hf_result.get('p99', 0):.4f}s")

        print("  Prefill-Only:")
        print(f"    Mean:     {prefill_time:.4f}s "
              f"({hf_time/prefill_time:.2f}x "
              f"{'faster' if prefill_time < hf_time else 'slower'})")
        print(f"    Median:   {prefill_result.get('median', 0):.4f}s")
        print(f"    P90:      {prefill_result.get('p90', 0):.4f}s")
        print(f"    P99:      {prefill_result.get('p99', 0):.4f}s")

        print("  Original:")
        print(f"    Mean:     {original_time:.4f}s "
              f"({hf_time/original_time:.2f}x "
              f"{'faster' if original_time < hf_time else 'slower'})")
        print(f"    Median:   {original_result.get('median', 0):.4f}s")
        print(f"    P90:      {original_result.get('p90', 0):.4f}s")
        print(f"    P99:      {original_result.get('p99', 0):.4f}s")

        print("  Prefill vs Original:")
        print(f"    Mean:     {original_time/prefill_time:.2f}x "
              f"{'faster' if prefill_time < original_time else 'slower'}")

        # Memory comparison
        print("\n Memory Comparison (Peak):")
        hf_mem = hf_result["peak_memory_mb"]
        prefill_mem = prefill_result["peak_memory_mb"]
        original_mem = original_result["peak_memory_mb"]

        print(f"  Transformers:        {hf_mem:.2f} MB")
        print(
            f"  Prefill-Only:        {prefill_mem:.2f} MB "
            f"({hf_mem/prefill_mem:.2f}x "
            f"{'less' if prefill_mem < hf_mem else 'more'})"
        )
        print(
            f"  Original:            {original_mem:.2f} MB "
            f"({hf_mem/original_mem:.2f}x "
            f"{'less' if original_mem < hf_mem else 'more'})"
        )
        prefill_vs_original_ratio = original_mem / prefill_mem
        prefill_vs_original_label = (
            "less" if prefill_mem < original_mem else "more"
        )
        print(
            f"  Prefill vs Original:  "
            f"{prefill_vs_original_ratio:.2f}x {prefill_vs_original_label}"
        )

    def _print_accuracy_comparison(self, acc_prefill, acc_original):
        """Print accuracy comparison (task-specific)."""
        raise NotImplementedError

    def run(self) -> Dict:
        """Run all tests and return results."""
        print("\n" + "=" * 80)
        print(f"Comprehensive {self.task_name} Test")
        print("=" * 80)

        results = {}

        # 1. Transformers baseline
        print("\n[1/3] Testing Transformers Baseline...")
        self.setup_test()
        try:
            results["transformers"] = self.test_transformers_baseline()
        finally:
            self.teardown_test()

        # 2. nano-vllm Prefill-Only
        print("\n[2/3] Testing nano-vllm Prefill-Only (Optimized)...")
        self.setup_test()
        try:
            results["nanovllm_prefill"] = self.test_nanovllm_prefill_only()
        finally:
            self.teardown_test()

        # 3. nano-vllm Original
        print("\n[3/3] Testing nano-vllm Original (Non-Optimized)...")
        self.setup_test()
        try:
            results["nanovllm_original"] = self.test_nanovllm_original()
        finally:
            self.teardown_test()

        # 4. Compare results
        print("\n[4/4] Comparison Results")
        print("-" * 80)
        self.compare_results(results)

        return results


class ComprehensiveRerankingTest(ComprehensiveTestBase):
    """Comprehensive reranking test."""

    task_type = "reranking"
    task_name = "Reranking"

    def __init__(
        self,
        model_path: str,
        queries: List[str],
        documents: List[str],
        reranker_type: str,
        is_listwise: bool,
        instruction: Optional[str] = None,
        num_warmup: int = 5,
        num_iterations: int = len(TEXT_RERANKING_QUERIES),
    ):
        super().__init__(model_path, num_warmup, num_iterations)
        self.queries = queries
        self.documents = documents
        self.query_doc_pairs = list(zip(queries, documents))
        self.instruction = instruction or (
            "Given a web search query, retrieve relevant passages "
            "that answer the query"
        )
        self.reranker_type = reranker_type
        self.is_listwise = is_listwise
        print(f"Using reranker type: {self.reranker_type} (listwise: {self.is_listwise})")

    def format_instruction(self, query: str, doc: str) -> str:
        """Format input for Qwen3-Reranker."""
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n<Document>: {doc}"
        )

    def get_prefix_suffix(self, tokenizer):
        """Get prefix and suffix tokens for Qwen3-Reranker."""
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the "
            "requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )
        suffix = (
            "<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        return prefix_tokens, suffix_tokens

    def test_transformers_baseline(self) -> Dict:
        """Test with transformers baseline."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_memory = get_memory_usage()

        if self.reranker_type == "jina_v3":
            # Listwise reranker: one query + multiple documents
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left", trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            
            # Load model using AutoModel
            # Get dtype from model config (matching nano-vllm)
            model_dtype = get_model_dtype(self.model_path)
            # Flash Attention 2 only supports fp16 and bf16
            # If dtype is float32, convert to float16
            if model_dtype == torch.float32:
                model_dtype = torch.float16
            model = AutoModel.from_pretrained(
                self.model_path,
                dtype=model_dtype,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            model = model.eval().cuda()
            
            # Format input: one query + all documents
            query = self.queries[0] if len(self.queries) > 0 else ""
            documents = self.documents
            
            # Use format function
            if hasattr(model, 'special_tokens'):
                prompt = format_docs_prompts_func(
                    query, documents, instruction=None,
                    special_tokens=model.special_tokens, no_thinking=True
                )
            else:
                prompt = format_docs_prompts_func(
                    query, documents, instruction=None, no_thinking=True
                )
            
            batch = tokenizer(
                text=[prompt],
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(model.device)
            
            def run_forward():
                with torch.no_grad():
                    outputs = model(**batch)
                    # For jina_v3, outputs.scores[0] contains all document scores
                    scores = outputs.scores[0].cpu().float().numpy()
                    return scores.tolist()
        
        elif self.reranker_type == "gemma":
            # Pointwise reranker: each query-document pair independently
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left"
            )
            # Get dtype from model config (matching nano-vllm)
            model_dtype = get_model_dtype(self.model_path)
            # Flash Attention 2 only supports fp16 and bf16
            # If dtype is float32, convert to float16
            if model_dtype == torch.float32:
                model_dtype = torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=model_dtype,
                attn_implementation="flash_attention_2",
            ).eval().cuda()
            
            yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
            
            pairs = list(zip(self.queries, self.documents))
            inputs = get_gemma_inputs(pairs, tokenizer)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            def run_forward():
                with torch.no_grad():
                    logits = model(**inputs, return_dict=True).logits
                    scores = logits[:, -1, yes_loc].view(-1, ).float().sigmoid()
                    return scores.cpu().tolist()
        
        else:
            # Default: Qwen3-Reranker (pointwise)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left", trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            # Get dtype from model config (matching nano-vllm)
            model_dtype = get_model_dtype(self.model_path)
            # Flash Attention 2 only supports fp16 and bf16
            # If dtype is float32, convert to float16
            if model_dtype == torch.float32:
                model_dtype = torch.float16
            model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
                .eval()
                .cuda()
            )

            token_false_id = tokenizer.convert_tokens_to_ids("no")
            token_true_id = tokenizer.convert_tokens_to_ids("yes")
            max_length = 8192

            prefix_tokens, suffix_tokens = self.get_prefix_suffix(tokenizer)

            pairs = [
                self.format_instruction(query, doc)
                for query, doc in self.query_doc_pairs
            ]

            # Process inputs
            inputs = tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
            )
            for i, ele in enumerate(inputs["input_ids"]):
                inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
            inputs = tokenizer.pad(
                inputs, padding=True, return_tensors="pt", max_length=max_length
            )
            for key in inputs:
                inputs[key] = inputs[key].to(model.device)

            def run_forward():
                with torch.no_grad():
                    outputs = model.model(**inputs)
                    hidden_states = (
                        outputs.last_hidden_state
                        if hasattr(outputs, "last_hidden_state")
                        else outputs[0]
                    )
                    logits = model.lm_head(hidden_states)

                    batch_scores = logits[:, -1, :]
                    true_vector = batch_scores[:, token_true_id]
                    false_vector = batch_scores[:, token_false_id]

                    batch_scores = torch.stack([false_vector, true_vector], dim=1)
                    batch_scores = torch.nn.functional.log_softmax(
                        batch_scores, dim=1
                    )
                    return batch_scores[:, 1].exp().tolist()

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_forward)
        scores = stats["result"]
        
        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory

        del model, tokenizer
        if 'inputs' in locals():
            del inputs
        if 'batch' in locals():
            del batch
        torch.cuda.empty_cache()

        return {
            "output": scores,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
        }

    def test_nanovllm_prefill_only(self) -> Dict:
        """Test with nano-vllm prefill-only mode."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_memory = get_memory_usage()

        if self.reranker_type == "jina_v3":
            # Listwise reranker: one query + multiple documents
            llm = LLM(
                self.model_path,
                is_reranker=True,
                reranker_type="jina_v3",
                projector_dim=512,
                use_flex_attention=False,
                enforce_eager=True,
                tensor_parallel_size=1,
                prefill_only_mode=True,  # Optimized version
            )
            
            # Reset peak memory stats after model loading
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Format input for jina_v3
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left", trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            
            query = self.queries[0] if len(self.queries) > 0 else ""
            documents = self.documents
            prompt = format_docs_prompts_func(query, documents, instruction=None)
            
            inputs = tokenizer(
                text=[prompt],
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            
            def run_rerank():
                # Call rerank method directly
                scores, query_embeds, doc_embeds = llm.model_runner.rerank(
                    input_ids, positions, attention_mask=attention_mask
                )
                # Convert to float32 before numpy conversion
                scores = scores.cpu().float().numpy()
                return scores.tolist()
        
        elif self.reranker_type == "gemma":
            # Pointwise reranker: each query-document pair independently
            llm = LLM(
                self.model_path,
                is_reranker=True,
                reranker_type="gemma",
                is_original_qwen3_reranker=True,
                classifier_from_token=["Yes"],
                enforce_eager=True,
                tensor_parallel_size=1,
                prefill_only_mode=True,  # Optimized version
            )
            
            # Reset peak memory stats after model loading
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Format inputs for gemma
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left"
            )
            pairs = list(zip(self.queries, self.documents))
            inputs = get_gemma_inputs(pairs, tokenizer)
            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            
            def run_rerank():
                # Call rerank method directly
                scores = llm.model_runner.rerank(
                    input_ids, positions, attention_mask=attention_mask
                )
                return scores.cpu().tolist()
        
        else:
            # Default: Qwen3-Reranker (pointwise)
            llm = LLM(
                self.model_path,
                is_reranker=True,
                reranker_type="qwen3",
                is_original_qwen3_reranker=True,
                classifier_from_token=["no", "yes"],
                enforce_eager=True,
                tensor_parallel_size=1,
                prefill_only_mode=True,  # Optimized version
            )
            
            # Reset peak memory stats after model loading
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            def run_rerank():
                scores = llm.rerank_batch(self.query_doc_pairs, use_tqdm=False)
                if isinstance(scores, tuple):
                    scores = scores[0]
                return scores.tolist() if hasattr(scores, "tolist") else scores

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_rerank)
        score_list = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory

        llm.exit()
        cleanup_distributed()

        return {
            "output": score_list,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
        }

    def test_nanovllm_original(self) -> Dict:
        """Test with nano-vllm original mode."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_memory = get_memory_usage()

        if self.reranker_type == "jina_v3":
            # Listwise reranker: one query + multiple documents
            llm = LLM(
                self.model_path,
                is_reranker=True,
                reranker_type="jina_v3",
                projector_dim=512,
                use_flex_attention=False,
                enforce_eager=True,
                tensor_parallel_size=1,
                prefill_only_mode=False,  # Original version
            )
            
            # Reset peak memory stats after model loading
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Format input for jina_v3
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left", trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            
            query = self.queries[0] if len(self.queries) > 0 else ""
            documents = self.documents
            prompt = format_docs_prompts_func(query, documents, instruction=None)
            
            inputs = tokenizer(
                text=[prompt],
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            
            def run_rerank():
                # Call rerank method directly
                scores, query_embeds, doc_embeds = llm.model_runner.rerank(
                    input_ids, positions, attention_mask=attention_mask
                )
                # Convert to float32 before numpy conversion
                scores = scores.cpu().float().numpy()
                return scores.tolist()
        
        elif self.reranker_type == "gemma":
            # Pointwise reranker: each query-document pair independently
            llm = LLM(
                self.model_path,
                is_reranker=True,
                reranker_type="gemma",
                is_original_qwen3_reranker=True,
                classifier_from_token=["Yes"],
                enforce_eager=True,
                tensor_parallel_size=1,
                prefill_only_mode=False,  # Original version
            )
            
            # Reset peak memory stats after model loading
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Format inputs for gemma
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left"
            )
            pairs = list(zip(self.queries, self.documents))
            inputs = get_gemma_inputs(pairs, tokenizer)
            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            
            def run_rerank():
                # Call rerank method directly
                scores = llm.model_runner.rerank(
                    input_ids, positions, attention_mask=attention_mask
                )
                return scores.cpu().tolist()
        
        else:
            # Default: Qwen3-Reranker (pointwise)
            llm = LLM(
                self.model_path,
                is_reranker=True,
                reranker_type="qwen3",
                is_original_qwen3_reranker=True,
                classifier_from_token=["no", "yes"],
                enforce_eager=True,
                tensor_parallel_size=1,
                prefill_only_mode=False,  # Original version
            )
            
            # Reset peak memory stats after model loading
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            def run_rerank():
                scores = llm.rerank_batch(self.query_doc_pairs, use_tqdm=False)
                if isinstance(scores, tuple):
                    scores = scores[0]
                return scores.tolist() if hasattr(scores, "tolist") else scores

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_rerank)
        score_list = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory

        llm.exit()
        cleanup_distributed()

        return {
            "output": score_list,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
        }

    def _print_accuracy_comparison(self, acc_prefill, acc_original):
        """Print accuracy comparison for reranking."""
        print("  Prefill-Only vs Transformers:")
        print(f"    Max Diff: {acc_prefill['max_diff']:.6f}")
        print(f"    Mean Diff: {acc_prefill['mean_diff']:.6f}")
        print(
            f"    Max Relative Error: "
            f"{acc_prefill['max_relative_error']:.6f}"
        )
        print(
            f"    Mean Relative Error: "
            f"{acc_prefill['mean_relative_error']:.6f}"
        )

        print("  Original vs Transformers:")
        print(f"    Max Diff: {acc_original['max_diff']:.6f}")
        print(f"    Mean Diff: {acc_original['mean_diff']:.6f}")
        print(
            f"    Max Relative Error: "
            f"{acc_original['max_relative_error']:.6f}"
        )
        print(
            f"    Mean Relative Error: "
            f"{acc_original['mean_relative_error']:.6f}"
        )


class ComprehensiveMultimodalRerankingTest(ComprehensiveTestBase):
    """Comprehensive multimodal reranking test."""

    task_type = "reranking"
    task_name = "Reranking (Multimodal)"

    def __init__(
        self,
        model_path: str,
        queries: List[str],
        documents: List[str],
        images: List[Image.Image],
        num_warmup: int = 5,
        num_iterations: int = len(MULTIMODAL_RERANKING_QUERIES),
    ):
        super().__init__(model_path, num_warmup, num_iterations)
        self.queries = queries
        self.documents = documents
        self.images = images
        self.query_doc_pairs = list(zip(queries, documents))

    def test_transformers_baseline(self) -> Dict:
        """Test with transformers baseline."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_memory = get_memory_usage()

        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # Set padding side to left for consistent last token extraction
        # This ensures the last token is the actual content token, not padding
        if hasattr(processor, 'tokenizer'):
            processor.tokenizer.padding_side = "left"
        # jina-reranker-m0 official usage: use AutoModel.from_pretrained
        # with trust_remote_code=True to load the custom JinaVLForRanking class
        # Get dtype from model config (matching nano-vllm)
        model_dtype = get_model_dtype(self.model_path)
        # Flash Attention 2 only supports fp16 and bf16
        # If dtype is float32, convert to float16
        if model_dtype == torch.float32:
            model_dtype = torch.float16
        model = AutoModel.from_pretrained(
            self.model_path,
            dtype=model_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        model.eval().cuda()

        # Format query-document pairs with images
        pairs = []
        for (query, doc), image in zip(self.query_doc_pairs, self.images):
            # Format as multimodal input
            formatted = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": f"Query: {query}\nDocument: {doc}"},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            pairs.append({"text": formatted, "images": [image]})

        # Reset peak memory stats after model loading, before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_forward():
            with torch.no_grad():
                # Batch process all pairs at once
                all_texts = [pair["text"] for pair in pairs]
                all_images = [pair["images"] for pair in pairs]
                
                processor_outputs = processor(
                    text=all_texts,
                    images=all_images,
                    return_tensors="pt",
                    padding=True,  # Processor will handle batch padding
                )
                inputs = {
                    k: v.to(model.device) for k, v in processor_outputs.items()
                }
                
                # Debug: print input info (only for first 2 pairs)
                for idx in range(min(2, len(pairs))):
                    print(f"\n[DEBUG Transformers Baseline] Pair {idx}:")
                    print(f"  input_ids shape: {inputs['input_ids'][idx:idx+1].shape}")
                    print(f"  input_ids (first 20): {inputs['input_ids'][idx, :20].tolist()}")
                    print(f"  input_ids (last 20): {inputs['input_ids'][idx, -20:].tolist()}")
                    if 'attention_mask' in inputs:
                        print(f"  attention_mask shape: {inputs['attention_mask'][idx:idx+1].shape}")
                        print(f"  attention_mask (first 20): {inputs['attention_mask'][idx, :20].tolist()}")
                        print(f"  attention_mask (last 20): {inputs['attention_mask'][idx, -20:].tolist()}")
                        print(f"  attention_mask sum: {inputs['attention_mask'][idx:idx+1].sum(dim=1).tolist()}")
                        # Check if padding is on left or right
                        first_nonzero = (
                            (inputs['attention_mask'][idx] != 0).nonzero(as_tuple=True)[0][0].item()
                            if (inputs['attention_mask'][idx] != 0).any() else -1
                        )
                        print(f"  first_nonzero (padding side check): {first_nonzero}")
                    if 'pixel_values' in inputs:
                        print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
                
                # For jina-reranker-m0, we need to get hidden_states and apply score head
                # The model should output hidden_states when output_hidden_states=True
                outputs = model(**inputs, output_hidden_states=True)
                
                # Extract hidden states from the last layer
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hidden_states = outputs.hidden_states[-1]  # Last layer [batch_size, seq_len, hidden_size]
                    
                    # Debug: print hidden states info (only for first 2 pairs)
                    for idx in range(min(2, len(pairs))):
                        print(f"\n[DEBUG Transformers Baseline] Pair {idx} hidden states:")
                        print(f"  hidden_states shape: {hidden_states[idx:idx+1].shape}")
                        print(f"  hidden_states[:, -1, :] mean: {hidden_states[idx, -1, :].mean().item():.6f}")
                        print(f"  hidden_states[:, -1, :] std: {hidden_states[idx, -1, :].std().item():.6f}")
                        print(f"  hidden_states[:, -1, :10]: {hidden_states[idx, -1, :10].tolist()}")
                    
                    # Get the last token's hidden state using attention_mask
                    # This matches JinaRerankerM0's logic: use attention_mask to find actual last token
                    if 'attention_mask' in inputs and inputs['attention_mask'] is not None:
                        attention_mask = inputs['attention_mask']
                        batch_size = hidden_states.shape[0]
                        # Find the actual last token for each sequence
                        last_token_indices = attention_mask.sum(dim=1) - 1  # [batch_size]
                        
                        # Debug: print last token info (only for first 2 pairs)
                        for idx in range(min(2, len(pairs))):
                            print(f"  Pair {idx} last_token_index: {last_token_indices[idx].item()}")
                        
                        # Extract hidden states at actual last token positions
                        last_token_hidden = hidden_states[
                            torch.arange(batch_size, device=hidden_states.device),
                            last_token_indices,
                            :
                        ]  # [batch_size, hidden_size]
                        
                        # Debug: print extracted last token hidden state (only for first 2 pairs)
                        for idx in range(min(2, len(pairs))):
                            print(f"  Pair {idx} last_token_hidden mean: {last_token_hidden[idx].mean().item():.6f}")
                            print(f"  Pair {idx} last_token_hidden std: {last_token_hidden[idx].std().item():.6f}")
                            print(f"  Pair {idx} last_token_hidden[:10]: {last_token_hidden[idx, :10].tolist()}")
                    else:
                        # If no attention_mask, assume left padding and use last token
                        last_token_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
                    
                    # Apply score head if available (jina-reranker-m0 has a score MLP)
                    if hasattr(model, 'score'):
                        # Use the model's score head
                        score_tensor = model.score(last_token_hidden)  # [batch_size, 1]
                        scores_list = score_tensor.squeeze(-1).tolist()  # [batch_size]
                        
                        # Debug: print scores (only for first 2 pairs)
                        for idx in range(min(2, len(pairs))):
                            print(f"  Pair {idx} score: {scores_list[idx]:.6f}")
                    else:
                        # Fallback: use mean of hidden states as score
                        scores_list = last_token_hidden.mean(dim=1).tolist()
                elif hasattr(outputs, 'logits'):
                    # Fallback: if no hidden_states, try logits
                    logits = outputs.logits
                    if 'attention_mask' in inputs and inputs['attention_mask'] is not None:
                        attention_mask = inputs['attention_mask']
                        batch_size = logits.shape[0]
                        last_token_indices = attention_mask.sum(dim=1) - 1
                        last_token_logits = logits[
                            torch.arange(batch_size, device=logits.device),
                            last_token_indices,
                        ]
                        scores_list = last_token_logits.mean(dim=1).tolist()
                    else:
                        scores_list = logits[:, -1, :].mean(dim=1).tolist()
                else:
                    # Last fallback: handle different output formats
                    if isinstance(outputs, tuple):
                        # If outputs is a tuple, use the first element
                        first_output = outputs[0]
                        if first_output.dim() == 1:
                            # 1D tensor: [batch_size] - already scores
                            scores_list = first_output.tolist()
                        elif first_output.dim() == 2:
                            # 2D tensor: [batch_size, hidden_size] - take mean
                            scores_list = first_output.mean(dim=1).tolist()
                        else:
                            # 3D or higher: [batch_size, seq_len, ...] - take last token and mean
                            scores_list = first_output[:, -1, :].mean(dim=1).tolist()
                    else:
                        # outputs is a tensor
                        if outputs.dim() == 1:
                            # 1D tensor: [batch_size] - already scores
                            scores_list = outputs.tolist()
                        elif outputs.dim() == 2:
                            # 2D tensor: [batch_size, hidden_size] - take mean
                            scores_list = outputs.mean(dim=1).tolist()
                        else:
                            # 3D or higher: [batch_size, seq_len, ...] - take last token and mean
                            scores_list = outputs[:, -1, :].mean(dim=1).tolist()
                
                return scores_list

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_forward)
        scores = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory

        del model, processor
        torch.cuda.empty_cache()

        return {
            "output": scores,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
        }

    def test_nanovllm_prefill_only(self) -> Dict:
        """Test with nano-vllm prefill-only mode."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_memory = get_memory_usage()

        llm = LLM(
            self.model_path,
            is_reranker=True,
            reranker_type="jina_m0",
            is_multimodal=True,
            enforce_eager=True,
            tensor_parallel_size=1,
            prefill_only_mode=True,
            trust_remote_code=True,
        )

        # Reset peak memory stats after model loading, before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_rerank():
            scores = llm.rerank_batch(
                self.query_doc_pairs,
                images=self.images,
                use_tqdm=False,
            )
            if isinstance(scores, tuple):
                scores = scores[0]
            return scores.tolist() if hasattr(scores, "tolist") else scores

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_rerank)
        score_list = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory

        llm.exit()
        cleanup_distributed()

        return {
            "output": score_list,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
        }

    def test_nanovllm_original(self) -> Dict:
        """Test with nano-vllm original mode."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_memory = get_memory_usage()

        llm = LLM(
            self.model_path,
            is_reranker=True,
            reranker_type="jina_m0",
            is_multimodal=True,
            enforce_eager=True,
            tensor_parallel_size=1,
            prefill_only_mode=False,
            trust_remote_code=True,
        )

        # Reset peak memory stats after model loading, before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_rerank():
            scores = llm.rerank_batch(
                self.query_doc_pairs,
                images=self.images,
                use_tqdm=False,
            )
            if isinstance(scores, tuple):
                scores = scores[0]
            return scores.tolist() if hasattr(scores, "tolist") else scores

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_rerank)
        score_list = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory

        llm.exit()
        cleanup_distributed()

        return {
            "output": score_list,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
        }

    def compare_results(self, results: Dict) -> None:
        """Compare and print results."""
        hf_result = results["transformers"]
        prefill_result = results["nanovllm_prefill"]
        original_result = results["nanovllm_original"]

        # Accuracy comparison
        acc_prefill = compare_results(
            hf_result.get("output", hf_result.get("result")),
            prefill_result.get("output", prefill_result.get("result")),
            task_type=self.task_type,
        )
        acc_original = compare_results(
            hf_result.get("output", hf_result.get("result")),
            original_result.get("output", original_result.get("result")),
            task_type=self.task_type,
        )

        self._print_accuracy_comparison(acc_prefill, acc_original)

        # Speed comparison
        self._print_speed_comparison(hf_result, prefill_result, original_result)

        # Memory comparison
        self._print_memory_comparison(hf_result, prefill_result, original_result)

    def _print_accuracy_comparison(self, acc_prefill, acc_original):
        """Print accuracy comparison for reranking."""
        print("  Prefill-Only vs Transformers:")
        print(f"    Max Diff: {acc_prefill['max_diff']:.6f}")
        print(f"    Mean Diff: {acc_prefill['mean_diff']:.6f}")
        print(
            f"    Max Relative Error: "
            f"{acc_prefill['max_relative_error']:.6f}"
        )
        print(
            f"    Mean Relative Error: "
            f"{acc_prefill['mean_relative_error']:.6f}"
        )

        print("  Original vs Transformers:")
        print(f"    Max Diff: {acc_original['max_diff']:.6f}")
        print(f"    Mean Diff: {acc_original['mean_diff']:.6f}")
        print(
            f"    Max Relative Error: "
            f"{acc_original['max_relative_error']:.6f}"
        )
        print(
            f"    Mean Relative Error: "
            f"{acc_original['mean_relative_error']:.6f}"
        )

    def _print_speed_comparison(self, hf_result, prefill_result, original_result):
        """Print speed comparison for reranking."""
        print("\n⚡ Speed Comparison:")
        hf_time = hf_result.get("mean", hf_result.get("time", 0))
        prefill_time = prefill_result.get(
            "mean", prefill_result.get("time", 0)
        )
        original_time = original_result.get(
            "mean", original_result.get("time", 0)
        )

        print("  Transformers:")
        print(f"    Mean:     {hf_time:.4f}s")
        print(f"    Median:   {hf_result.get('median', 0):.4f}s")
        print(f"    P90:      {hf_result.get('p90', 0):.4f}s")
        print(f"    P99:      {hf_result.get('p99', 0):.4f}s")

        print("  Prefill-Only:")
        print(f"    Mean:     {prefill_time:.4f}s "
              f"({hf_time/prefill_time:.2f}x "
              f"{'faster' if prefill_time < hf_time else 'slower'})")
        print(f"    Median:   {prefill_result.get('median', 0):.4f}s")
        print(f"    P90:      {prefill_result.get('p90', 0):.4f}s")
        print(f"    P99:      {prefill_result.get('p99', 0):.4f}s")

        print("  Original:")
        print(f"    Mean:     {original_time:.4f}s "
              f"({hf_time/original_time:.2f}x "
              f"{'faster' if original_time < hf_time else 'slower'})")
        print(f"    Median:   {original_result.get('median', 0):.4f}s")
        print(f"    P90:      {original_result.get('p90', 0):.4f}s")
        print(f"    P99:      {original_result.get('p99', 0):.4f}s")

        print("  Prefill vs Original:")
        print(f"    Mean:     {original_time/prefill_time:.2f}x "
              f"{'faster' if prefill_time < original_time else 'slower'}")

    def _print_memory_comparison(self, hf_result, prefill_result, original_result):
        """Print memory comparison for reranking."""
        print("\n💾 Memory Comparison (Peak):")
        hf_mem = hf_result["peak_memory_mb"]
        prefill_mem = prefill_result["peak_memory_mb"]
        original_mem = original_result["peak_memory_mb"]

        print(f"  Transformers:        {hf_mem:.2f} MB")
        print(
            f"  Prefill-Only:        {prefill_mem:.2f} MB "
            f"({hf_mem/prefill_mem:.2f}x "
            f"{'less' if prefill_mem < hf_mem else 'more'})"
        )
        print(
            f"  Original:            {original_mem:.2f} MB "
            f"({hf_mem/original_mem:.2f}x "
            f"{'less' if original_mem < hf_mem else 'more'})"
        )
        prefill_vs_original_ratio = original_mem / prefill_mem
        prefill_vs_original_label = (
            "less" if prefill_mem < original_mem else "more"
        )
        print(
            f"  Prefill vs Original:  "
            f"{prefill_vs_original_ratio:.2f}x {prefill_vs_original_label}"
        )


def test_reranking_comprehensive(
    modality: str = "text",
    num_warmup: int = 5,
    num_iterations: int = len(TEXT_RERANKING_QUERIES),
    batch_size: Optional[int] = None,
    model_path: Optional[str] = None,
    model: Optional[str] = None,
):
    """Comprehensive reranking test using class-based approach.
    
    Args:
        modality: "text" or "multimodal"
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        batch_size: Optional batch size for testing
    """
    if modality == "text":
        # Text-only reranking test
        if model_path is None:
            raise ValueError("--model-path is required for text reranking test")
        model_path = os.path.expanduser(model_path)

        # Base queries and documents from prompt.py
        base_queries = list(TEXT_RERANKING_QUERIES)
        base_documents = list(TEXT_RERANKING_DOCUMENTS)

        # Get reranker type from model name (required)
        if model is None:
            raise ValueError("--model is required for text reranking test")
        reranker_type, is_listwise = get_reranker_type_from_model(model)
        print(f"Using reranker type: {reranker_type} (listwise: {is_listwise})")
        
        if is_listwise:
            # For listwise reranker (jina_v3): one query + multiple documents
            # Use first query and all documents
            query = base_queries[0] if len(base_queries) > 0 else "What is the capital of China?"
            documents = base_documents
            
            # Scale up documents if batch_size is specified
            if batch_size and batch_size > len(documents):
                documents = []
                for i in range(batch_size):
                    documents.append(base_documents[i % len(base_documents)])
            
            queries = [query]  # Single query for listwise
            if batch_size:
                print(f"Testing with 1 query and {len(documents)} documents (listwise)")
        else:
            # For pointwise reranker: multiple query-document pairs
            # Scale up batch size if specified
            if batch_size and batch_size > len(base_queries):
                # Repeat and vary the queries/documents to reach desired batch size
                queries = []
                documents = []
                for i in range(batch_size):
                    queries.append(base_queries[i % len(base_queries)])
                    documents.append(base_documents[i % len(base_documents)])
            else:
                queries = base_queries
                documents = base_documents
            
            if batch_size:
                print(f"Testing with batch size: {len(queries)} (pointwise)")

        try:
            test = ComprehensiveRerankingTest(
                model_path,
                queries,
                documents,
                reranker_type,
                is_listwise,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
            )
            return test.run()
        except FileNotFoundError:
            print(f"Model not found at {model_path}")
            print("Skipping comprehensive reranking test")
            return None
        except Exception as e:
            print(f"Error in comprehensive reranking test: {e}")
            import traceback
            traceback.print_exc()
            return None

    elif modality == "multimodal":
        # Multimodal reranking test - test jina-reranker-m0
        if model_path is None:
            raise ValueError("--model-path is required for multimodal reranking test")
        model_path = os.path.expanduser(model_path)

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Skipping multimodal reranking test")
            return None

        # Get reranker type from model name (required)
        if model is None:
            raise ValueError("--model is required for multimodal reranking test")
        reranker_type, is_listwise = get_reranker_type_from_model(model)
        print(f"Using reranker type: {reranker_type} (listwise: {is_listwise})")
        
        # Filter by model name if specified
        if not should_test_model(model):
            return None

        print(f"\n{'=' * 80}")
        print(f"Testing {model} (Multimodal)")
        print(f"{'=' * 80}")

        # Download test images
        print("Downloading test images...")
        images = [download_image(url) for url in DEFAULT_IMAGE_URLS[:10]]

        base_queries = list(MULTIMODAL_RERANKING_QUERIES)

        base_documents = [
            "A dog playing in the park",
            "A beautiful sunset over the beach",
            "A city skyline at night",
        ]

        # Scale up batch size if specified
        if batch_size and batch_size > len(base_queries):
            queries = []
            documents = []
            image_urls = DEFAULT_IMAGE_URLS * ((batch_size // len(DEFAULT_IMAGE_URLS)) + 1)
            images = [download_image(url) for url in image_urls[:batch_size]]
            variations_q = [
                "What is in this image?",
                "Describe the scene.",
                "What objects can you see?",
                "What colors are present?",
                "What is the main subject?",
            ]
            variations_d = [
                "A dog playing in the park",
                "A beautiful sunset over the beach",
                "A city skyline at night",
                "A cat sitting on a window",
                "A mountain landscape",
            ]
            for i in range(batch_size):
                queries.append(variations_q[i % len(variations_q)])
                documents.append(variations_d[i % len(variations_d)])
        else:
            queries = base_queries
            documents = base_documents

        if batch_size:
            print(f"Testing with batch size: {len(queries)}")

        try:
            test = ComprehensiveMultimodalRerankingTest(
                model_path,
                queries,
                documents,
                images[:len(queries)],
                num_warmup=num_warmup,
                num_iterations=num_iterations,
            )
            return test.run()
        except FileNotFoundError:
            print(f"Model not found at {model_path}")
            print("Skipping multimodal reranking test")
            return None
        except Exception as e:
            print(f"Error in multimodal reranking test: {e}")
            import traceback
            traceback.print_exc()
            return None

    else:
        raise ValueError(f"Unknown modality: {modality}")



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test prefill-only reranking optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test text-only reranking
  python test_prefill_only_rerank.py --modality text

  # Test multimodal reranking
  python test_prefill_only_rerank.py --modality multimodal

  # Test specific model
  python test_prefill_only_rerank.py --modality multimodal --model jina-reranker-m0

  # Test with custom batch size
  python test_prefill_only_rerank.py --modality text --batch-size 32

  # Test with custom warmup and iterations
  python test_prefill_only_rerank.py --modality all \\
      --num-warmup 10 --num-iterations 20 --batch-size 16
        """,
    )

    parser.add_argument(
        "--modality",
        type=str,
        choices=["text", "multimodal", "all"],
        default="all",
        help="Modality type to test (default: all)",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to test (required). Examples: 'qwen3-reranker', "
        "'jina-reranker-v3', 'bge-reranker-v2-gemma', 'jina-reranker-m0'.",
    )

    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )

    parser.add_argument(
        "--num-iterations",
        type=int,
        default=len(TEXT_RERANKING_QUERIES),
        help=f"Number of benchmark iterations (default: {len(TEXT_RERANKING_QUERIES)}, from prompt.py)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for testing. If not specified, uses default "
        "batch size for each test. For comprehensive tests, this will "
        "scale the number of test samples.",
    )

    parser.add_argument(
        "--memory",
        action="store_true",
        help="Run memory comparison tests",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory (required).",
    )

    return parser.parse_args()


def main():
    """Run reranking tests based on command line arguments."""
    args = parse_args()

    print("=" * 60)
    print("Prefill-Only Reranking Test Suite")
    print("=" * 60)
    print(f"Modality: {args.modality}")
    if args.model:
        print(f"Model: {args.model}")
        set_model_filter(args.model)
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, tests may be slow")

    # Determine default iterations based on modality
    default_text_iterations = len(TEXT_RERANKING_QUERIES)
    default_multimodal_iterations = len(MULTIMODAL_RERANKING_QUERIES)
    
    # If user is using the default value (from parse_args), adjust it based on modality
    if args.num_iterations == default_text_iterations:
        if args.modality == "multimodal":
            args.num_iterations = default_multimodal_iterations
        elif args.modality == "all":
            # For "all", keep text default (will be adjusted per test)
            pass

    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Benchmark iterations: {args.num_iterations}")

    # Comprehensive tests only
    print("\n" + "=" * 80)
    print("Comprehensive Tests")
    print("=" * 80)

    if args.modality in ["text", "all"]:
        try:
            text_iterations = args.num_iterations
            if args.modality == "all" and args.num_iterations == default_text_iterations:
                text_iterations = default_text_iterations
            test_reranking_comprehensive(
                modality="text",
                num_warmup=args.num_warmup,
                num_iterations=text_iterations,
                batch_size=args.batch_size,
                model_path=args.model_path,
                model=args.model,
            )
        except Exception as e:
            print(f"Error in comprehensive reranking test (text): {e}")
            import traceback
            traceback.print_exc()

    if args.modality in ["multimodal", "all"]:
        try:
            multimodal_iterations = args.num_iterations
            if args.modality == "all" and args.num_iterations == default_text_iterations:
                multimodal_iterations = default_multimodal_iterations
            test_reranking_comprehensive(
                modality="multimodal",
                num_warmup=args.num_warmup,
                num_iterations=multimodal_iterations,
                batch_size=args.batch_size,
                model_path=args.model_path,
                model=args.model,
            )
        except Exception as e:
            print(f"Error in comprehensive reranking test (multimodal): {e}")
            import traceback
            traceback.print_exc()

    # Memory comparison
    if args.memory:
        print("\n" + "=" * 80)
        print("Memory Comparison Tests")
        print("=" * 80)
        try:
            test_memory_comparison(model_path=args.model_path)
        except Exception as e:
            print(f"Error in memory comparison test: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
