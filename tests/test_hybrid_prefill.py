"""Test hybrid prefilling correctness and memory savings.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/test_hybrid_prefill.py

Tests:
1. Numerical equivalence: chunked MLP == full MLP (bitwise identical)
2. Peak memory benchmark at various sequence lengths
"""
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def init_dist():
    """Minimal single-GPU dist init for model construction."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def test_correctness():
    """Verify chunked MLP produces bit-identical results to full MLP."""
    from nanovllm.layers.hybrid_prefill import chunked_mlp_forward, set_hybrid_prefill_config

    print("=" * 60)
    print("TEST 1: Numerical Correctness (MLP only)")
    print("=" * 60)
    print("  (MLP is the only layer being chunked — attention is unchanged)")
    print()

    from transformers import AutoConfig
    from nanovllm.models.qwen3 import Qwen3MLP

    model_path = "/mnt/nas-tbt/zhouxinxu/llm/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(model_path)

    mlp = Qwen3MLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=getattr(config, "hidden_act", "silu"),
    ).cuda().eval().to(torch.bfloat16)

    test_cases = [
        (1024, 2048, "short seq (1K), chunk=2048 (passthrough)"),
        (8192, 2048, "8K tokens, chunk=2048"),
        (16384, 4096, "16K tokens, chunk=4096"),
        (32768, 4096, "32K tokens, chunk=4096"),
        (65536, 8192, "64K tokens (long video), chunk=8192"),
        (76800, 4096, "76.8K tokens (Qwen3-VL video), chunk=4096"),
    ]

    all_passed = True
    for seq_len, chunk_size, desc in test_cases:
        torch.manual_seed(42)
        hidden_states = torch.randn(seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        # Full MLP forward (no chunking)
        with torch.no_grad():
            out_full = mlp(hidden_states)

        # Chunked MLP forward
        with torch.no_grad():
            out_chunk = chunked_mlp_forward(hidden_states, mlp, chunk_size=chunk_size)

        # Compare — should be bit-identical since we just split along token dim
        max_diff = (out_full - out_chunk).abs().max().item()
        passed = max_diff == 0.0

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}: max_diff={max_diff:.2e}")
        if not passed:
            all_passed = False

    set_hybrid_prefill_config(False)
    print(f"\n  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print()
    return all_passed


def benchmark_memory():
    """Benchmark peak GPU memory for MLP pass with and without chunking.

    Simulates what happens inside each decoder layer's MLP. With N layers,
    the total memory savings multiply by N (each layer benefits independently).
    """
    from nanovllm.layers.hybrid_prefill import chunked_mlp_forward, set_hybrid_prefill_config

    print("=" * 60)
    print("TEST 2: Peak Memory Benchmark (per-layer MLP activation)")
    print("=" * 60)

    from transformers import AutoConfig
    from nanovllm.models.qwen3 import Qwen3MLP

    model_path = "/mnt/nas-tbt/zhouxinxu/llm/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(model_path)

    mlp = Qwen3MLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=getattr(config, "hidden_act", "silu"),
    ).cuda().eval().to(torch.bfloat16)

    param_mem = sum(p.numel() * p.element_size() for p in mlp.parameters()) / 1024**2
    print(f"  MLP params: {param_mem:.1f} MB (hidden={config.hidden_size}, intermediate={config.intermediate_size})")
    print(f"  Model has {config.num_hidden_layers} layers — each layer saves independently")
    print()

    seq_lengths = [4096, 8192, 16384, 32768, 65536, 76800]
    chunk_size = 4096

    print(f"  Chunk size: {chunk_size}")
    print(f"  {'SeqLen':>8} | {'Normal (MB)':>12} | {'Hybrid (MB)':>12} | {'Saved (MB)':>11} | {'Saved %':>8}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*11}-+-{'-'*8}")

    for seq_len in seq_lengths:
        results = {}
        for mode in ["normal", "hybrid"]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated() / 1024**2

            torch.manual_seed(42)
            hidden_states = torch.randn(seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

            with torch.no_grad():
                if mode == "normal":
                    _ = mlp(hidden_states)
                else:
                    _ = chunked_mlp_forward(hidden_states, mlp, chunk_size=chunk_size)

            peak_mb = torch.cuda.max_memory_allocated() / 1024**2 - base_mem
            results[mode] = peak_mb

            del hidden_states, _
            torch.cuda.empty_cache()

        saved = results["normal"] - results["hybrid"]
        pct = saved / results["normal"] * 100 if results["normal"] > 0 else 0
        print(f"  {seq_len:>8} | {results['normal']:>10.1f}   | {results['hybrid']:>10.1f}   | {saved:>9.1f}   | {pct:>6.1f}%")

    set_hybrid_prefill_config(False)
    print()


def benchmark_speed():
    """Benchmark latency impact of chunked MLP (per-layer).

    Tests MLP forward pass directly since full model forward requires
    engine context (KV cache, cu_seqlens) that isn't available standalone.
    """
    from nanovllm.layers.hybrid_prefill import chunked_mlp_forward

    print("=" * 60)
    print("TEST 3: Latency Impact (per-layer MLP)")
    print("=" * 60)

    from transformers import AutoConfig
    from nanovllm.models.qwen3 import Qwen3MLP

    model_path = "/mnt/nas-tbt/zhouxinxu/llm/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(model_path)

    mlp = Qwen3MLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=getattr(config, "hidden_act", "silu"),
    ).cuda().eval().to(torch.bfloat16)

    seq_lengths = [8192, 32768, 65536]
    chunk_size = 4096
    warmup = 3
    repeats = 10

    print(f"  Chunk size: {chunk_size}, Warmup: {warmup}, Repeats: {repeats}")
    print(f"  {'SeqLen':>8} | {'Normal (ms)':>12} | {'Hybrid (ms)':>12} | {'Overhead':>10}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    for seq_len in seq_lengths:
        results = {}
        for mode in ["normal", "hybrid"]:
            torch.manual_seed(42)
            hidden_states = torch.randn(seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    if mode == "normal":
                        _ = mlp(hidden_states)
                    else:
                        _ = chunked_mlp_forward(hidden_states, mlp, chunk_size=chunk_size)
                torch.cuda.synchronize()

            # Timed runs
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                with torch.no_grad():
                    if mode == "normal":
                        _ = mlp(hidden_states)
                    else:
                        _ = chunked_mlp_forward(hidden_states, mlp, chunk_size=chunk_size)
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / repeats * 1000
            results[mode] = elapsed

            del hidden_states, _
            torch.cuda.empty_cache()

        overhead_pct = (results["hybrid"] - results["normal"]) / results["normal"] * 100
        print(f"  {seq_len:>8} | {results['normal']:>10.2f}   | {results['hybrid']:>10.2f}   | {overhead_pct:>+8.1f}%")

    print()


if __name__ == "__main__":
    init_dist()
    torch.cuda.set_device(0)

    test_correctness()
    benchmark_memory()
    benchmark_speed()

    dist.destroy_process_group()
    print("Done.")
