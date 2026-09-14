"""Hybrid prefilling: chunked MLP correctness and benchmarks.

`chunked_mlp_forward` splits the MLP forward pass along the token dimension,
bounding peak activation memory from `seq_len * intermediate_size` down to
`chunk_size * intermediate_size`. Only the MLP is chunked — attention still
sees the full sequence — so the output must be identical to the plain MLP.

The correctness test needs GPU + local weights and is marked `slow`.
Run with:
    pytest tests/test_hybrid_prefill.py -m slow -v
"""
import os
import time

import pytest

torch = pytest.importorskip("torch")

from tests.conftest import model_path, requires_cuda

pytestmark = pytest.mark.slow

# (seq_len, chunk_size) pairs covering the passthrough case and long sequences.
CORRECTNESS_CASES = [
    (1024, 2048),
    (8192, 2048),
    (16384, 4096),
    (32768, 4096),
    (65536, 8192),
    (76800, 4096),
]


@pytest.fixture
def single_process_group():
    """The parallel-linear layers need a process group even for world_size=1."""
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        yield
        dist.destroy_process_group()
    else:
        yield


def _make_mlp():
    """Build a random-weight Qwen3MLP on GPU; returns (mlp, hidden_size)."""
    from transformers import AutoConfig

    from nanovllm.models.qwen3 import Qwen3MLP

    path = model_path("qwen3")
    config = AutoConfig.from_pretrained(path)
    mlp = Qwen3MLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=getattr(config, "hidden_act", "silu"),
    ).cuda().to(torch.bfloat16).eval()
    # Random but finite weights (the constructor leaves them uninitialised).
    torch.manual_seed(0)
    with torch.no_grad():
        for p in mlp.parameters():
            p.normal_(0.0, 0.02)
    return mlp, config.hidden_size


def test_chunked_mlp_is_identical(single_process_group):
    """Chunked MLP must be bit-identical to the full MLP for every case."""
    requires_cuda()
    from nanovllm.layers.hybrid_prefill import chunked_mlp_forward

    mlp, hidden_size = _make_mlp()

    for seq_len, chunk_size in CORRECTNESS_CASES:
        torch.manual_seed(42)
        x = torch.randn(seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            out_full = mlp(x)
            out_chunk = chunked_mlp_forward(x, mlp, chunk_size=chunk_size)
        max_diff = (out_full - out_chunk).abs().max().item()
        assert max_diff == 0.0, (
            f"seq_len={seq_len} chunk_size={chunk_size}: max_diff={max_diff:.2e}"
        )


def _benchmark_memory(mlp, hidden_size):
    """Peak GPU memory for one MLP pass with and without chunking (not a test)."""
    from nanovllm.layers.hybrid_prefill import chunked_mlp_forward

    print("  Chunk size: 4096")
    print(f"  {'SeqLen':>8} | {'Normal (MB)':>12} | {'Hybrid (MB)':>12} | {'Saved (MB)':>11} | {'Saved %':>8}")
    for seq_len in [4096, 8192, 16384, 32768, 65536, 76800]:
        results = {}
        for mode in ["normal", "hybrid"]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated() / 1024**2
            torch.manual_seed(42)
            x = torch.randn(seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                if mode == "normal":
                    _ = mlp(x)
                else:
                    _ = chunked_mlp_forward(x, mlp, chunk_size=4096)
            results[mode] = torch.cuda.max_memory_allocated() / 1024**2 - base_mem
            del x, _
            torch.cuda.empty_cache()
        saved = results["normal"] - results["hybrid"]
        pct = saved / results["normal"] * 100 if results["normal"] > 0 else 0
        print(f"  {seq_len:>8} | {results['normal']:>10.1f}   | {results['hybrid']:>10.1f}   | {saved:>9.1f}   | {pct:>6.1f}%")


def _benchmark_speed(mlp, hidden_size):
    """Latency impact of chunked MLP (not a test)."""
    from nanovllm.layers.hybrid_prefill import chunked_mlp_forward

    print("  Chunk size: 4096, Warmup: 3, Repeats: 10")
    print(f"  {'SeqLen':>8} | {'Normal (ms)':>12} | {'Hybrid (ms)':>12} | {'Overhead':>10}")
    for seq_len in [8192, 32768, 65536]:
        results = {}
        for mode in ["normal", "hybrid"]:
            torch.manual_seed(42)
            x = torch.randn(seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                with torch.inference_mode():
                    if mode == "normal":
                        _ = mlp(x)
                    else:
                        _ = chunked_mlp_forward(x, mlp, chunk_size=4096)
                torch.cuda.synchronize()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                with torch.inference_mode():
                    if mode == "normal":
                        _ = mlp(x)
                    else:
                        _ = chunked_mlp_forward(x, mlp, chunk_size=4096)
                torch.cuda.synchronize()
            results[mode] = (time.perf_counter() - t0) / 10 * 1000
            del x, _
            torch.cuda.empty_cache()
        overhead_pct = (results["hybrid"] - results["normal"]) / results["normal"] * 100
        print(f"  {seq_len:>8} | {results['normal']:>10.2f}   | {results['hybrid']:>10.2f}   | {overhead_pct:>+8.1f}%")


if __name__ == "__main__":
    """Standalone run: correctness + memory/speed benchmarks."""
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(0)

    try:
        from nanovllm.layers.hybrid_prefill import chunked_mlp_forward

        mlp, hidden_size = _make_mlp()
        for seq_len, chunk_size in CORRECTNESS_CASES:
            torch.manual_seed(42)
            x = torch.randn(seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                out_full = mlp(x)
                out_chunk = chunked_mlp_forward(x, mlp, chunk_size=chunk_size)
            max_diff = (out_full - out_chunk).abs().max().item()
            assert max_diff == 0.0, f"seq_len={seq_len} chunk_size={chunk_size}"
        print("correctness: PASSED")

        _benchmark_memory(mlp, hidden_size)
        _benchmark_speed(mlp, hidden_size)
    finally:
        dist.destroy_process_group()
