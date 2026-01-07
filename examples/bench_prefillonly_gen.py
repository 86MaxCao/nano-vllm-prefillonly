"""Comprehensive test script for prefill-only generation optimizations.

This script provides comprehensive comparisons for single-token
generation tasks:
1. Accuracy comparison with transformers baseline
2. Speed comparison with transformers baseline
3. Memory usage comparison with transformers baseline
4. Speed comparison: prefill-only vs original nano-vllm
5. Memory comparison: prefill-only vs original nano-vllm

Models tested:
- Text-only: Qwen3-0.6B
- Multimodal: llavanext, qwen2vl, qwen2.5vl, qwen3vl

Usage:
    python bench_prefillonly_gen.py --modality text
    python bench_prefillonly_gen.py --modality multimodal
    python bench_prefillonly_gen.py --modality all --model qwen2vl
    python bench_prefillonly_gen.py --batch-size 32 --num-warmup 10
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
# Import specific multimodal model classes
try:
    from transformers import (
        LlavaNextForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )
except ImportError:
    # Fallback if specific imports are not available
    LlavaNextForConditionalGeneration = None
    Qwen2VLForConditionalGeneration = None
    Qwen2_5_VLForConditionalGeneration = None
    Qwen3VLForConditionalGeneration = None

from nanovllm import LLM, SamplingParams

# ============================================================================
# Test Configuration: Image URLs and Prompts
# ============================================================================
# Test images from COCO (10 images for comprehensive testing)
DEFAULT_IMAGE_URLS = (
    "http://images.cocodataset.org/val2017/000000000285.jpg",
    "http://images.cocodataset.org/val2017/000000000632.jpg",
    "http://images.cocodataset.org/val2017/000000000724.jpg",
    "http://images.cocodataset.org/val2017/000000000776.jpg",
    "http://images.cocodataset.org/val2017/000000001000.jpg",
    "http://images.cocodataset.org/val2017/000000001268.jpg",
    "http://images.cocodataset.org/val2017/000000006012.jpg",
    "http://images.cocodataset.org/val2017/000000190236.jpg",
    "http://images.cocodataset.org/val2017/000000331352.jpg",
    "http://images.cocodataset.org/val2017/000000517069.jpg",
)

# Test prompts for multimodal generation (10 prompts matching the 10 images)
# Each prompt expects a one-word answer
MULTIMODAL_GENERATION_PROMPTS = (
    "What animal is in the picture? Answer with one word.",  # Expected: bear
    "Was the image taken indoors or outdoors? Answer with indoors or outdoors.",  # Expected: indoors
    "Was the image taken outdoors? Answer with yes or no.",  # Expected: yes
    "How many bears in the picture? Answer with one word.",  # Expected: 3
    "Was the image taken outdoors? Answer with yes or no.",  # Expected: yes
    "Was the image taken indoors or outdoors? Answer with indoors or outdoors.",  # Expected: outdoors
    "How many kinds of fruits are in the picture? Answer with one word.",  # Expected: 2
    "Was the image taken indoors or outdoors? Answer with indoors or outdoors.",  # Expected: indoors
    "Was the image taken indoors or outdoors? Answer with indoors or outdoors.",  # Expected: indoors
    "What's the color of the building? Answer with one word.",  # Expected: white
)

# Global variable to store model filter
_MODEL_FILTER = None


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
        "qwen2.5vl": ["qwen2.5vl", "qwen2_5_vl", "qwen2.5-vl", "qwen2-5vl"],
        "qwen3vl": ["qwen3vl", "qwen3_vl", "qwen3-vl"],
        "llavanext": ["llavanext", "llava-v1.6", "llava_v1.6"],
    }
    
    if filter_lower in model_name_mappings:
        return any(
            m in model_lower for m in model_name_mappings[filter_lower]
        )
    return filter_lower in model_lower


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


def get_detailed_memory_stats():
    """Get detailed GPU memory statistics for analysis.
    
    Returns a dictionary with comprehensive memory metrics including:
    - allocated: Currently allocated memory
    - reserved: Memory reserved by PyTorch allocator
    - active: Active memory (in use)
    - inactive: Inactive memory (fragmented)
    - peak_allocated: Peak allocated memory
    - peak_reserved: Peak reserved memory
    - fragmentation: Memory fragmentation ratio
    """
    if not torch.cuda.is_available():
        return {}
    
    stats = torch.cuda.memory_stats()
    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    active = stats.get("active_bytes.all.current", 0)
    inactive = stats.get("inactive_split_bytes.all.current", 0)
    peak_allocated = stats.get("allocated_bytes.all.peak", 0)
    peak_reserved = stats.get("reserved_bytes.all.peak", 0)
    
    # Calculate fragmentation if reserved > 0
    fragmentation = (inactive / reserved * 100) if reserved > 0 else 0.0
    
    return {
        "allocated_mb": allocated / 1024**2,
        "reserved_mb": reserved / 1024**2,
        "active_mb": active / 1024**2,
        "inactive_mb": inactive / 1024**2,
        "peak_allocated_mb": peak_allocated / 1024**2,
        "peak_reserved_mb": peak_reserved / 1024**2,
        "fragmentation_percent": fragmentation,
    }


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
        enforce_eager=True,
        tensor_parallel_size=1,
        prefill_only_mode=True,
        single_token_mode=True,
    )

    prompts = ["Is the capital of China Beijing? Answer with Yes or No."] * 10
    _ = llm_prefill.generate_single_token(
        prompts,
        SamplingParams(temperature=1e-6, max_tokens=1),
        use_tqdm=False,
    )

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
        enforce_eager=True,
        tensor_parallel_size=1,
        prefill_only_mode=False,
        single_token_mode=True,
    )

    _ = llm_original.generate_single_token(
        prompts,
        SamplingParams(temperature=1e-6, max_tokens=1),
        use_tqdm=False,
    )

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

        # Get actual prediction results
        hf_output = hf_result.get("output", hf_result.get("result"))
        prefill_output = prefill_result.get("output", prefill_result.get("result"))
        original_output = original_result.get("output", original_result.get("result"))

        # Accuracy comparison
        print("\n📊 Accuracy Comparison:")
        acc_prefill = compare_results(
            hf_output,
            prefill_output,
            task_type=self.task_type,
        )
        acc_original = compare_results(
            hf_output,
            original_output,
            task_type=self.task_type,
        )

        # Get tokenizer/processor for detailed output
        tokenizer = getattr(self, '_tokenizer', None)
        processor = getattr(self, '_processor', None)
        tokenizer_to_use = processor.tokenizer if processor else tokenizer

        self._print_accuracy_comparison(
            acc_prefill, 
            acc_original,
            hf_output=hf_output,
            prefill_output=prefill_output,
            original_output=original_output,
            tokenizer=tokenizer_to_use,
        )

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
        
        # Detailed memory statistics
        hf_stats = hf_result.get("detailed_stats", {})
        prefill_stats = prefill_result.get("detailed_stats", {})
        original_stats = original_result.get("detailed_stats", {})
        
        if hf_stats or prefill_stats or original_stats:
            print("\n Detailed Memory Statistics:")
            print("  " + "-" * 76)
            print(f"  {'Metric':<20} {'Transformers':<18} {'Prefill-Only':<18} {'Original':<18}")
            print("  " + "-" * 76)
            
            metrics = [
                ("Peak Allocated", "peak_allocated_mb"),
                ("Peak Reserved", "peak_reserved_mb"),
                ("Current Allocated", "allocated_mb"),
                ("Current Reserved", "reserved_mb"),
                ("Active", "active_mb"),
                ("Inactive (Frag)", "inactive_mb"),
                ("Fragmentation %", "fragmentation_percent"),
            ]
            
            for metric_name, metric_key in metrics:
                hf_val = hf_stats.get(metric_key, 0)
                prefill_val = prefill_stats.get(metric_key, 0)
                original_val = original_stats.get(metric_key, 0)
                
                if metric_key == "fragmentation_percent":
                    print(
                        f"  {metric_name:<20} "
                        f"{hf_val:>6.2f}%{'':<10} "
                        f"{prefill_val:>6.2f}%{'':<10} "
                        f"{original_val:>6.2f}%"
                    )
                else:
                    print(
                        f"  {metric_name:<20} "
                        f"{hf_val:>8.2f} MB{'':<8} "
                        f"{prefill_val:>8.2f} MB{'':<8} "
                        f"{original_val:>8.2f} MB"
                    )
            print("  " + "-" * 76)

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


class ComprehensiveGenerationTest(ComprehensiveTestBase):
    """Comprehensive single token generation test."""

    task_type = "generation"
    task_name = "Single Token Generation"

    def __init__(
        self,
        model_path: str,
        prompts: List[str],
        num_warmup: int = 5,
        num_iterations: int = 10,
    ):
        super().__init__(model_path, num_warmup, num_iterations)
        self.prompts = prompts

    def test_transformers_baseline(self) -> Dict:
        """Test with transformers baseline."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_memory = get_memory_usage()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # Save tokenizer for detailed output later
        self._tokenizer = tokenizer
        model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            .eval()
            .cuda()
        )

        # Tokenize prompts
        inputs = tokenizer(
            self.prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048,
        ).to(model.device)

        # Reset peak memory stats after model loading, before inference
        # This ensures we only measure inference peak, not model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Set seed for reproducibility (matching nano-vllm)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        def run_forward():
            with torch.no_grad():
                outputs = model(**inputs)
                # Handle different output formats
                # For CausalLMOutput, use outputs.logits
                # For ModelOutput, try outputs.logits or get from hidden_states/last_hidden_state
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 'last_hidden_state'):
                    # If no logits, compute from last_hidden_state using lm_head
                    hidden_states = outputs.last_hidden_state
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    # Some models return hidden_states as a tuple/list, get the last one
                    hidden_states = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, (list, tuple)) else outputs.hidden_states
                else:
                    raise AttributeError(f"Model output has no 'logits', 'last_hidden_state', or 'hidden_states': {type(outputs)}, available attributes: {dir(outputs)}")
                
                # If we got hidden_states, compute logits using lm_head
                if 'hidden_states' in locals():
                    if hasattr(model, 'lm_head'):
                        logits = model.lm_head(hidden_states)
                    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
                        logits = model.language_model.lm_head(hidden_states)
                    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                        # Some models have the structure: model.model -> language_model -> lm_head
                        if hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'lm_head'):
                            logits = model.model.language_model.lm_head(hidden_states)
                        else:
                            raise AttributeError(f"Cannot find lm_head in model to compute logits from hidden_states: {type(model)}")
                    else:
                        raise AttributeError(f"Cannot find lm_head in model to compute logits from hidden_states: {type(model)}")
                
                # Get first token after prompt (single token generation)
                next_token_logits = logits[:, -1, :]
                # Sample with temperature (use same seed for reproducibility)
                next_token_ids = torch.multinomial(
                    torch.softmax(next_token_logits / 1e-6, dim=-1), 1
                ).squeeze(-1)
                return next_token_ids.cpu().tolist()

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_forward)
        token_ids = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory
        detailed_stats = get_detailed_memory_stats()

        del model, tokenizer, inputs
        torch.cuda.empty_cache()

        return {
            "output": token_ids,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
            "detailed_stats": detailed_stats,
        }

    def test_nanovllm_prefill_only(self) -> Dict:
        """Test with nano-vllm prefill-only mode."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_memory = get_memory_usage()

        llm = LLM(
            self.model_path,
            enforce_eager=True,
            tensor_parallel_size=1,
            prefill_only_mode=True,  # Optimized version
            single_token_mode=True,
            seed=0,  # Set seed for reproducibility
        )
        
        # Reset peak memory stats after model loading, before inference
        # This ensures we only measure inference peak, not model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_generate():
            return llm.generate_single_token(
                self.prompts,
                SamplingParams(temperature=1e-6, max_tokens=1),
                use_tqdm=False,
            )

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_generate)
        token_ids = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory
        detailed_stats = get_detailed_memory_stats()

        llm.exit()
        cleanup_distributed()

        return {
            "output": token_ids,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
            "detailed_stats": detailed_stats,
        }

    def test_nanovllm_original(self) -> Dict:
        """Test with nano-vllm original mode."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_memory = get_memory_usage()

        llm = LLM(
            self.model_path,
            enforce_eager=True,
            tensor_parallel_size=1,
            prefill_only_mode=False,  # Original version
            single_token_mode=True,
            seed=0,  # Set seed for reproducibility
        )
        
        # Reset peak memory stats after model loading, before inference
        # This ensures we only measure inference peak, not model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_generate():
            return llm.generate_single_token(
                self.prompts,
                SamplingParams(temperature=1e-6, max_tokens=1),
                use_tqdm=False,
            )

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_generate)
        token_ids = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory
        detailed_stats = get_detailed_memory_stats()

        llm.exit()
        cleanup_distributed()

        return {
            "output": token_ids,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
            "detailed_stats": detailed_stats,
        }

    def _print_accuracy_comparison(
        self, 
        acc_prefill, 
        acc_original,
        hf_output=None,
        prefill_output=None,
        original_output=None,
        tokenizer=None,
    ):
        """Print accuracy comparison for generation."""
        print("  Prefill-Only vs Transformers:")
        print(
            f"    Exact Matches: {acc_prefill['exact_matches']}/"
            f"{acc_prefill['total']}"
        )
        print(f"    Match Rate: {acc_prefill['match_rate']:.2%}")

        print("  Original vs Transformers:")
        print(
            f"    Exact Matches: {acc_original['exact_matches']}/"
            f"{acc_original['total']}"
        )
        print(f"    Match Rate: {acc_original['match_rate']:.2%}")

        # Print detailed results if tokenizer is available
        if tokenizer is not None and hf_output is not None and prefill_output is not None:
            print("\n  📋 Detailed Results (Prefill-Only vs Transformers):")
            prompts = getattr(self, 'prompts', [])
            for i, (hf_token, nv_token) in enumerate(zip(hf_output, prefill_output)):
                match = "✓" if hf_token == nv_token else "✗"
                prompt_preview = prompts[i][:60] + "..." if i < len(prompts) and len(prompts[i]) > 60 else (prompts[i] if i < len(prompts) else f"Prompt {i+1}")
                try:
                    # Handle tensor conversion
                    if isinstance(hf_token, torch.Tensor):
                        hf_token_val = hf_token.item()
                    else:
                        hf_token_val = int(hf_token)
                    if isinstance(nv_token, torch.Tensor):
                        nv_token_val = nv_token.item()
                    else:
                        nv_token_val = int(nv_token)
                    
                    hf_text = tokenizer.decode([hf_token_val], skip_special_tokens=True)
                    nv_text = tokenizer.decode([nv_token_val], skip_special_tokens=True)
                    
                    print(f"    [{i+1:2d}] {match} {prompt_preview}")
                    print(f"        Transformers: token_id={hf_token_val:6d} -> '{hf_text}'")
                    print(f"        Prefill-Only:  token_id={nv_token_val:6d} -> '{nv_text}'")
                except Exception as e:
                    # Fallback if decoding fails
                    print(f"    [{i+1:2d}] {match} {prompt_preview}")
                    print(f"        Transformers: token_id={hf_token}")
                    print(f"        Prefill-Only:  token_id={nv_token}")
                    print(f"        (Decode error: {e})")


class ComprehensiveMultimodalGenerationTest(ComprehensiveTestBase):
    """Comprehensive single token generation test for multimodal models."""

    task_type = "generation"
    task_name = "Single Token Generation (Multimodal)"

    def __init__(
        self,
        model_path: str,
        model_type: str,
        prompts: List[str],
        images: List[Image.Image],
        num_warmup: int = 5,
        num_iterations: int = 10,
    ):
        super().__init__(model_path, num_warmup, num_iterations)
        self.model_type = model_type
        self.prompts = prompts
        self.images = images

    def test_transformers_baseline(self) -> Dict:
        """Test with transformers baseline."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_memory = get_memory_usage()

        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # Save processor for detailed output later
        self._processor = processor
        
        # Load multimodal models using their specific classes
        # Reference: nano-vllm-multimodal/test_hf_qwen3vl.py
        model = None
        if self.model_type == "llavanext" and LlavaNextForConditionalGeneration:
            try:
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
            except Exception:
                pass
        elif self.model_type == "qwen2_vl" and Qwen2VLForConditionalGeneration:
            try:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
            except Exception:
                pass
        elif self.model_type == "qwen2_5_vl" and Qwen2_5_VLForConditionalGeneration:
            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
            except Exception:
                pass
        elif self.model_type == "qwen3_vl" and Qwen3VLForConditionalGeneration:
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
            except Exception:
                pass
        
        # Fallback to AutoModel if specific class loading failed
        if model is None:
            try:
                model = AutoModel.from_pretrained(
                    self.model_path,
                    dtype=torch.float16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
            except (ValueError, TypeError, ImportError):
                # Final fallback to AutoModelForCausalLM for text-only models
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype=torch.float16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
        
        if not torch.cuda.is_available():
            model = model.to("cpu")
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        # Format requests for multimodal generation
        requests = []
        for prompt, image in zip(self.prompts, self.images):
            chat_prompt = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            requests.append({"text": chat_prompt, "images": [image]})

        # Reset peak memory stats after model loading, before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Set seed for reproducibility
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        def run_forward():
            with torch.no_grad():
                token_ids_list = []
                for req in requests:
                    processor_outputs = processor(
                        text=[req["text"]],
                        images=req["images"],
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {
                        k: v.to(model.device) for k, v in processor_outputs.items()
                    }
                    outputs = model(**inputs)
                    # Handle different output formats
                    # For CausalLMOutput, use outputs.logits
                    # For ModelOutput, try outputs.logits or get from hidden_states/last_hidden_state
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif hasattr(outputs, 'last_hidden_state'):
                        # If no logits, compute from last_hidden_state using lm_head
                        hidden_states = outputs.last_hidden_state
                    elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        # Some models return hidden_states as a tuple/list, get the last one
                        hidden_states = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, (list, tuple)) else outputs.hidden_states
                    else:
                        raise AttributeError(f"Model output has no 'logits', 'last_hidden_state', or 'hidden_states': {type(outputs)}, available attributes: {dir(outputs)}")
                    
                    # If we got hidden_states, compute logits using lm_head
                    if 'hidden_states' in locals():
                        if hasattr(model, 'lm_head'):
                            logits = model.lm_head(hidden_states)
                        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
                            logits = model.language_model.lm_head(hidden_states)
                        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                            # Some models have the structure: model.model -> language_model -> lm_head
                            if hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'lm_head'):
                                logits = model.model.language_model.lm_head(hidden_states)
                            else:
                                raise AttributeError(f"Cannot find lm_head in model to compute logits from hidden_states: {type(model)}")
                        else:
                            raise AttributeError(f"Cannot find lm_head in model to compute logits from hidden_states: {type(model)}")
                    
                    next_token_logits = logits[:, -1, :]
                    next_token_ids = torch.multinomial(
                        torch.softmax(next_token_logits / 1e-6, dim=-1), 1
                    ).squeeze(-1)
                    token_ids_list.append(next_token_ids.cpu().item())
                return token_ids_list

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_forward)
        token_ids = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory
        detailed_stats = get_detailed_memory_stats()

        # Don't delete processor, we need it for detailed output
        del model
        torch.cuda.empty_cache()

        return {
            "output": token_ids,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
            "detailed_stats": detailed_stats,
        }

    def test_nanovllm_prefill_only(self) -> Dict:
        """Test with nano-vllm prefill-only mode."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_memory = get_memory_usage()

        llm = LLM(
            self.model_path,
            is_multimodal=True,
            multimodal_model_type=self.model_type,
            enforce_eager=True,
            tensor_parallel_size=1,
            prefill_only_mode=True,
            single_token_mode=True,
            seed=0,
            trust_remote_code=True,
        )

        # Reset peak memory stats after model loading, before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # Save processor for detailed output later
        self._processor = processor

        # Format requests
        requests = []
        for prompt, image in zip(self.prompts, self.images):
            chat_prompt = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            requests.append({"text": chat_prompt, "images": [image]})

        def run_generate():
            outputs = llm.generate_multimodal(
                requests,
                SamplingParams(temperature=1e-6, max_tokens=1),
                processor,
                use_tqdm=False,
            )
            # Extract token IDs from outputs
            # generate_multimodal returns dict with "text" and "token_ids" fields
            token_ids = []
            for i, out in enumerate(outputs):
                # Directly use token_ids from output (more accurate than re-tokenizing text)
                if "token_ids" in out and out["token_ids"]:
                    # token_ids is a list, get the first (and only) token for single-token generation
                    token_id = out["token_ids"][0] if isinstance(out["token_ids"], list) else out["token_ids"]
                    token_ids.append(token_id)
                else:
                    # Fallback: try to extract from text (should not happen in normal case)
                    text = out.get("text", "").strip()
                    if text:
                        print(f"[WARNING test_nanovllm_prefill_only] Output {i} missing token_ids, "
                              f"falling back to text tokenization: '{text}'")
                        tokens = processor.tokenizer.encode(text, add_special_tokens=False)
                        token_ids.append(tokens[0] if tokens else 0)
                    else:
                        print(f"[WARNING test_nanovllm_prefill_only] Output {i} has no token_ids or text")
                        token_ids.append(0)
            return token_ids

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_generate)
        token_ids = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory
        detailed_stats = get_detailed_memory_stats()

        llm.exit()
        cleanup_distributed()

        return {
            "output": token_ids,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
            "detailed_stats": detailed_stats,
        }

    def test_nanovllm_original(self) -> Dict:
        """Test with nano-vllm original mode."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_memory = get_memory_usage()

        llm = LLM(
            self.model_path,
            is_multimodal=True,
            multimodal_model_type=self.model_type,
            enforce_eager=True,
            tensor_parallel_size=1,
            prefill_only_mode=False,
            single_token_mode=True,
            seed=0,
            trust_remote_code=True,
        )

        # Reset peak memory stats after model loading, before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # Save processor for detailed output later
        self._processor = processor

        # Format requests
        requests = []
        for prompt, image in zip(self.prompts, self.images):
            chat_prompt = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            requests.append({"text": chat_prompt, "images": [image]})

        def run_generate():
            outputs = llm.generate_multimodal(
                requests,
                SamplingParams(temperature=1e-6, max_tokens=1),
                processor,
                use_tqdm=False,
            )
            # Extract token IDs from outputs
            # generate_multimodal returns dict with "text" and "token_ids" fields
            token_ids = []
            for i, out in enumerate(outputs):
                # Directly use token_ids from output (more accurate than re-tokenizing text)
                if "token_ids" in out and out["token_ids"]:
                    # token_ids is a list, get the first (and only) token for single-token generation
                    token_id = out["token_ids"][0] if isinstance(out["token_ids"], list) else out["token_ids"]
                    token_ids.append(token_id)
                else:
                    # Fallback: try to extract from text (should not happen in normal case)
                    text = out.get("text", "").strip()
                    if text:
                        print(f"[WARNING test_nanovllm_original] Output {i} missing token_ids, "
                              f"falling back to text tokenization: '{text}'")
                        tokens = processor.tokenizer.encode(text, add_special_tokens=False)
                        token_ids.append(tokens[0] if tokens else 0)
                    else:
                        print(f"[WARNING test_nanovllm_original] Output {i} has no token_ids or text")
                        token_ids.append(0)
            return token_ids

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_generate)
        token_ids = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory
        detailed_stats = get_detailed_memory_stats()

        llm.exit()
        cleanup_distributed()

        return {
            "output": token_ids,
            "mean": stats["mean"],
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "peak_memory_mb": peak_memory,
            "memory_mb": memory,
            "detailed_stats": detailed_stats,
        }

    def _print_accuracy_comparison(
        self, 
        acc_prefill, 
        acc_original,
        hf_output=None,
        prefill_output=None,
        original_output=None,
        tokenizer=None,
    ):
        """Print accuracy comparison for generation."""
        print("  Prefill-Only vs Transformers:")
        print(
            f"    Exact Matches: {acc_prefill['exact_matches']}/"
            f"{acc_prefill['total']}"
        )
        print(f"    Match Rate: {acc_prefill['match_rate']:.2%}")

        print("  Original vs Transformers:")
        print(
            f"    Exact Matches: {acc_original['exact_matches']}/"
            f"{acc_original['total']}"
        )
        print(f"    Match Rate: {acc_original['match_rate']:.2%}")

        # Print detailed results if tokenizer is available
        if tokenizer is not None and hf_output is not None and prefill_output is not None:
            print("\n  📋 Detailed Results (Prefill-Only vs Transformers):")
            prompts = getattr(self, 'prompts', [])
            for i, (hf_token, nv_token) in enumerate(zip(hf_output, prefill_output)):
                match = "✓" if hf_token == nv_token else "✗"
                prompt_preview = prompts[i][:60] + "..." if i < len(prompts) and len(prompts[i]) > 60 else (prompts[i] if i < len(prompts) else f"Prompt {i+1}")
                try:
                    # Handle tensor conversion
                    if isinstance(hf_token, torch.Tensor):
                        hf_token_val = hf_token.item()
                    else:
                        hf_token_val = int(hf_token)
                    if isinstance(nv_token, torch.Tensor):
                        nv_token_val = nv_token.item()
                    else:
                        nv_token_val = int(nv_token)
                    
                    hf_text = tokenizer.decode([hf_token_val], skip_special_tokens=True)
                    nv_text = tokenizer.decode([nv_token_val], skip_special_tokens=True)
                    
                    print(f"    [{i+1:2d}] {match} {prompt_preview}")
                    print(f"        Transformers: token_id={hf_token_val:6d} -> '{hf_text}'")
                    print(f"        Prefill-Only:  token_id={nv_token_val:6d} -> '{nv_text}'")
                except Exception as e:
                    # Fallback if decoding fails
                    print(f"    [{i+1:2d}] {match} {prompt_preview}")
                    print(f"        Transformers: token_id={hf_token}")
                    print(f"        Prefill-Only:  token_id={nv_token}")
                    print(f"        (Decode error: {e})")


def test_generation_comprehensive(
    modality: str = "text",
    num_warmup: int = 5,
    num_iterations: int = 10,
    batch_size: Optional[int] = None,
    model_path: Optional[str] = None,
    model: Optional[str] = None,
):
    """Comprehensive generation test using class-based approach.
    
    Args:
        modality: "text" or "multimodal"
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        batch_size: Optional batch size for testing
    """
    if modality == "text":
        # Text-only generation test
        if model_path is None:
            raise ValueError("--model-path is required for text generation test")
        model_path = os.path.expanduser(model_path)

        # Base prompts
        base_prompts = [
            "Is the capital of China Beijing? Answer with Yes or No.",
            "Is the capital of France London? Answer with Yes or No.",
            "Is 2+2 equal to 4? Answer with Yes or No.",
        ]

        # Scale up batch size if specified
        if batch_size and batch_size > len(base_prompts):
            # Repeat and vary the prompts to reach desired batch size
            prompts = []
            variations = [
                "Is the capital of China Beijing? Answer with Yes or No.",
                "Is the capital of France London? Answer with Yes or No.",
                "Is 2+2 equal to 4? Answer with Yes or No.",
                "Is water made of H2O? Answer with Yes or No.",
                "Is the Earth round? Answer with Yes or No.",
                "Is Python a programming language? Answer with Yes or No.",
                "Is the sun a star? Answer with Yes or No.",
                "Is 3+3 equal to 6? Answer with Yes or No.",
            ]
            for i in range(batch_size):
                prompts.append(variations[i % len(variations)])
        else:
            prompts = base_prompts

        if batch_size:
            print(f"Testing with batch size: {len(prompts)}")

        try:
            test = ComprehensiveGenerationTest(
                model_path,
                prompts,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
            )
            return test.run()
        except FileNotFoundError:
            print(f"Model not found at {model_path}")
            print("Skipping comprehensive generation test")
            return None
        except Exception as e:
            print(f"Error in comprehensive generation test: {e}")
            import traceback
            traceback.print_exc()
            return None

    elif modality == "multimodal":
        # Multimodal generation test - test single VLM model
        if model_path is None:
            raise ValueError("--model-path is required for multimodal generation test")
        if model is None:
            raise ValueError("--model is required for multimodal generation test")
        
        model_path = os.path.expanduser(model_path)
        
        # Map model name to model type
        model_lower = model.lower()
        model_type_mapping = {
            "llavanext": "llavanext",
            "qwen2vl": "qwen2_vl",
            "qwen2.5vl": "qwen2_5_vl",
            "qwen3vl": "qwen3_vl",
        }
        
        # Find matching model type
        model_type = None
        for name, mtype in model_type_mapping.items():
            if name in model_lower:
                model_type = mtype
                break
        
        if model_type is None:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Supported models: llavanext, qwen2vl, qwen2.5vl, qwen3vl"
            )
        
        vlm_models = [(model, model_type, model_path)]

        # Download test images
        print("Downloading test images...")
        images = [download_image(url) for url in DEFAULT_IMAGE_URLS[:10]]

        base_prompts = list(MULTIMODAL_GENERATION_PROMPTS)

        # Scale up batch size if specified
        if batch_size and batch_size > len(base_prompts):
            prompts = []
            image_urls = DEFAULT_IMAGE_URLS * ((batch_size // len(DEFAULT_IMAGE_URLS)) + 1)
            images = [download_image(url) for url in image_urls[:batch_size]]
            variations = [
                "Is there a dog in this image? Answer with Yes or No.",
                "Is there a cat in this image? Answer with Yes or No.",
                "Is there a person in this image? Answer with Yes or No.",
                "Is there a car in this image? Answer with Yes or No.",
                "Is there a bird in this image? Answer with Yes or No.",
            ]
            for i in range(batch_size):
                prompts.append(variations[i % len(variations)])
        else:
            prompts = base_prompts

        if batch_size:
            print(f"Testing with batch size: {len(prompts)}")

        results = {}
        for model_name, model_type, model_path in vlm_models:
            # Filter by model name if specified
            if not should_test_model(model_name):
                continue

            model_path = os.path.expanduser(model_path)
            if not os.path.exists(model_path):
                print(f"\n{model_name} not found at {model_path}, skipping...")
                continue

            print(f"\n{'=' * 80}")
            print(f"Testing {model_name}")
            print(f"{'=' * 80}")

            try:
                test = ComprehensiveMultimodalGenerationTest(
                    model_path,
                    model_type,
                    prompts,
                    images[:len(prompts)],
                    num_warmup=num_warmup,
                    num_iterations=num_iterations,
                )
                results[model_name] = test.run()
            except FileNotFoundError:
                print(f"Model not found at {model_path}")
                print(f"Skipping {model_name}")
                continue
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return results

    else:
        raise ValueError(f"Unknown modality: {modality}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test prefill-only generation optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test text-only generation
  python test_prefill_only_gen.py --modality text \\
      --model qwen3 \\
      --model-path ~/.cache/huggingface/hub/Qwen3-0.6B

  # Test multimodal generation
  python test_prefill_only_gen.py --modality multimodal \\
      --model llavanext \\
      --model-path ~/.cache/huggingface/hub/llava-v1.6-mistral-7b-hf

  # Test with custom batch size
  python test_prefill_only_gen.py --modality text \\
      --model qwen3 \\
      --model-path ~/.cache/huggingface/hub/Qwen3-0.6B \\
      --batch-size 32

  # Test with custom warmup and iterations
  python test_prefill_only_gen.py --modality text \\
      --model qwen3 \\
      --model-path ~/.cache/huggingface/hub/Qwen3-0.6B \\
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
        help="Model to test (required). Examples: 'qwen3' (text), "
        "'llavanext', 'qwen2vl', 'qwen2.5vl', 'qwen3vl' (multimodal).",
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
        default=10,
        help="Number of benchmark iterations (default: 10)",
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
    """Run generation tests based on command line arguments."""
    args = parse_args()

    print("=" * 60)
    print("Prefill-Only Generation Test Suite")
    print("=" * 60)
    print(f"Modality: {args.modality}")
    print(f"Model: {args.model}")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, tests may be slow")

    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Benchmark iterations: {args.num_iterations}")

    print("\n" + "=" * 80)
    print("Comprehensive Tests")
    print("=" * 80)

    if args.modality in ["text", "all"]:
        try:
            test_generation_comprehensive(
                modality="text",
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
                batch_size=args.batch_size,
                model_path=args.model_path,
                model=args.model,
            )
        except Exception as e:
            print(f"Error in comprehensive generation test (text): {e}")
            import traceback
            traceback.print_exc()

    if args.modality in ["multimodal", "all"]:
        try:
            test_generation_comprehensive(
                modality="multimodal",
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
                batch_size=args.batch_size,
                model_path=args.model_path,
                model=args.model,
            )
        except Exception as e:
            print(f"Error in comprehensive generation test (multimodal): {e}")
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
