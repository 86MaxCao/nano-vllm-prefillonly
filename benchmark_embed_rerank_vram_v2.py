"""
Benchmark: VRAM savings for embedding/reranker models.

Uses subprocess isolation to get accurate memory measurements.
Compares:
  - Prefill-only mode (no KV cache) — our optimization
  - With KV cache — simulates vLLM's behavior (allocates KV cache even for embed/rerank)

For models that can't be loaded as generation models, we compute KV cache size
from the same formula the framework uses and add it to the model weights VRAM.
"""
import os
import json
import subprocess
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU_ID", "2")
os.environ["MASTER_ADDR"] = "127.0.0.1"

PYTHON = "/mnt/nas-tbt/caoziqi/micromamba/envs/main/bin/python"

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub/hf_cache")

MODELS = {
    "qwen3-embedding-0.6b": {
        "path": f"{HF_CACHE}/Qwen3-Embedding-0.6B",
        "task": "embed",
        "texts": ["What is the capital of China?", "Explain gravity"],
        "prefill_kwargs": {
            "is_embedding": True,
            "pooling_type": "LAST", "normalize_embeddings": True,
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
    "bge-gemma2": {
        "path": f"{HF_CACHE}/bge-multilingual-gemma2",
        "task": "embed",
        "texts": ["What is the capital of China?", "Explain gravity"],
        "prefill_kwargs": {
            "is_embedding": True,
            "pooling_type": "LAST", "normalize_embeddings": True,
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
    "qwen3-reranker-0.6b": {
        "path": f"{HF_CACHE}/Qwen3-Reranker-0.6B",
        "task": "rerank",
        "pairs": [("What is the capital of China?", "The capital of China is Beijing.")],
        "prefill_kwargs": {
            "is_reranker": True,
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
    "jina-reranker-v3": {
        "path": f"{HF_CACHE}/jina-reranker-v3",
        "task": "rerank",
        "pairs": [("What is the capital of China?", "The capital of China is Beijing.")],
        "prefill_kwargs": {
            "is_reranker": True,
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
    "bge-reranker-gemma": {
        "path": f"{HF_CACHE}/bge-reranker-v2-gemma",
        "task": "rerank",
        "pairs": [("What is the capital of China?", "The capital of China is Beijing.")],
        "prefill_kwargs": {
            "is_reranker": True,
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
    "qwen3-vl-embedding-2b": {
        "path": f"{HF_CACHE}/Qwen3-VL-Embedding-2B",
        "task": "embed",
        "texts": ["What is the capital of China?", "Explain gravity"],
        "prefill_kwargs": {
            "is_embedding": True, "embedding_type": "qwen3_vl",
            "pooling_type": "LAST", "normalize_embeddings": True,
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
    "qwen3-vl-reranker-2b": {
        "path": f"{HF_CACHE}/Qwen3-VL-Reranker-2B",
        "task": "rerank",
        "pairs": [("What is the capital of China?", "The capital of China is Beijing.")],
        "prefill_kwargs": {
            "is_reranker": True, "reranker_type": "qwen3_vl",
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
    "jina-reranker-m0": {
        "path": f"{HF_CACHE}/jina-reranker-m0",
        "task": "rerank",
        "pairs": [("What is the capital of China?", "The capital of China is Beijing.")],
        "prefill_kwargs": {
            "is_reranker": True, "reranker_type": "jina_m0",
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
}


def run_isolated(name, cfg, with_kvcache):
    """Run a single measurement in an isolated subprocess."""
    result_file = os.path.join(tempfile.gettempdir(), f"vram_{name}_{int(with_kvcache)}.json")

    if with_kvcache:
        # Load as generation model — forces KV cache allocation
        kwargs = {"enforce_eager": True, "tensor_parallel_size": 1}
        # Run inference? No — just measure after loading + KV cache
    else:
        kwargs = cfg["prefill_kwargs"]

    script = f'''
import json, os, gc, torch, random
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(random.randint(30000, 60000))
os.environ["CUDA_VISIBLE_DEVICES"] = "{os.environ['CUDA_VISIBLE_DEVICES']}"

from nanovllm import LLM

model_path = "{cfg['path']}"
kwargs = {repr(kwargs)}
with_kvcache = {int(with_kvcache)}

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

try:
    llm = LLM(model_path, **kwargs)
    torch.cuda.synchronize()

    # Run inference only for prefill mode to ensure model is fully loaded
    if not with_kvcache:
        if "{cfg['task']}" == "embed":
            llm.embed_batch({repr(cfg.get('texts', ['test']))}, use_tqdm=False)
        else:
            llm.rerank_batch({repr(cfg.get('pairs', [('q', 'd')]))}, use_tqdm=False)
        torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    current_mb = torch.cuda.memory_allocated() / 1024**2

    mr = llm.model_runner
    kvcache_mb = 0
    if mr.kv_cache is not None:
        kvcache_mb = mr.kv_cache.numel() * mr.kv_cache.element_size() / 1024**2

    # Also get model architecture info for manual KV cache calculation
    hf_config = mr.config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    num_hidden_layers = getattr(text_config, "num_hidden_layers", getattr(hf_config, "num_hidden_layers", 0))
    num_kv_heads = getattr(text_config, "num_key_value_heads", getattr(text_config, "num_attention_heads", 0))
    num_attention_heads = getattr(text_config, "num_attention_heads", getattr(hf_config, "num_attention_heads", 0))
    hidden_size = getattr(text_config, "hidden_size", getattr(hf_config, "hidden_size", 0))
    head_dim = getattr(text_config, "head_dim", hidden_size // num_attention_heads if num_attention_heads > 0 else 128)
    block_size = getattr(mr, "block_size", 16)
    gpu_memory_utilization = getattr(mr.config, "gpu_memory_utilization", 0.9)

    llm.exit()
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    result = {{
        "peak_mb": peak_mb,
        "current_mb": current_mb,
        "kvcache_mb": kvcache_mb,
        "num_blocks": mr.config.num_kvcache_blocks,
        "num_hidden_layers": num_hidden_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "ok": True,
    }}
except Exception as e:
    import traceback
    result = {{"ok": False, "error": traceback.format_exc()}}

with open("{result_file}", "w") as f:
    json.dump(result, f)
'''

    script_path = os.path.join(tempfile.gettempdir(), f"vram_script_{name}_{int(with_kvcache)}.py")
    with open(script_path, "w") as f:
        f.write(script)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = project_dir + ":" + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [PYTHON, script_path],
            env=env, capture_output=True, timeout=300,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")[-1000:]
            return {"ok": False, "error": f"exit={proc.returncode}: {stderr}"}
        with open(result_file) as f:
            return json.load(f)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def compute_kvcache_size_mb(num_hidden_layers, num_kv_heads, head_dim, block_size,
                            gpu_total_mb, model_weights_mb, gpu_memory_utilization=0.9,
                            dtype_bytes=2, is_multimodal=False):
    """Compute KV cache size using the same formula as model_runner.allocate_kv_cache."""
    block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype_bytes
    # available_memory = total * utilization - model_weights
    available_bytes = gpu_total_mb * 1024**2 * gpu_memory_utilization - model_weights_mb * 1024**2
    if is_multimodal:
        available_bytes *= 0.5
    num_blocks = int(available_bytes) // block_bytes
    kvcache_bytes = num_blocks * block_bytes
    return kvcache_bytes / 1024**2, num_blocks


def main():
    results = []

    for name, cfg in MODELS.items():
        if not os.path.exists(cfg["path"]):
            print(f"SKIP: {name}")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")

        # Step 1: Measure prefill-only (no KV cache) VRAM
        print(f"  [1/2] Prefill-Only (no KV cache) ...")
        r1 = run_isolated(name, cfg, with_kvcache=False)
        if r1["ok"]:
            print(f"    Peak: {r1['peak_mb']:.0f} MB, KV cache: {r1['kvcache_mb']:.0f} MB, blocks: {r1['num_blocks']}")
        else:
            print(f"    FAILED: {r1.get('error', 'unknown')[:200]}")

        # Step 2: Try loading with KV cache (generation mode)
        print(f"  [2/2] With KV cache (simulating vLLM) ...")
        r2 = run_isolated(name, cfg, with_kvcache=True)
        if r2["ok"]:
            print(f"    Peak: {r2['peak_mb']:.0f} MB, KV cache: {r2['kvcache_mb']:.0f} MB, blocks: {r2['num_blocks']}")
            with_kvcache_peak = r2["peak_mb"]
            kvcache_mb = r2["kvcache_mb"]
        else:
            print(f"    Generation mode failed: {str(r2.get('error', ''))[:100]}")
            # Compute KV cache size from model architecture
            if r1["ok"] and r1["num_hidden_layers"] > 0:
                import torch
                free, total = torch.cuda.mem_get_info()
                total_mb = total / 1024**2
                # Model weights = current_mb after loading (peak includes temporary allocations)
                model_weights_mb = r1["peak_mb"]
                computed_kv_mb, computed_blocks = compute_kvcache_size_mb(
                    r1["num_hidden_layers"], r1["num_kv_heads"], r1["head_dim"],
                    r1["block_size"], total_mb, model_weights_mb,
                    r1["gpu_memory_utilization"]
                )
                with_kvcache_peak = model_weights_mb + computed_kv_mb
                kvcache_mb = computed_kv_mb
                print(f"    Computed: model={model_weights_mb:.0f}MB + KV_cache={computed_kv_mb:.0f}MB = {with_kvcache_peak:.0f}MB (blocks={computed_blocks})")
                r2 = {"ok": True, "peak_mb": with_kvcache_peak, "kvcache_mb": kvcache_mb}
            else:
                r2 = {"ok": False}

        if r1["ok"] and r2["ok"]:
            saved_pct = (1 - r1["peak_mb"] / r2["peak_mb"]) * 100
            results.append({
                "name": name,
                "no_kvcache_mb": r1["peak_mb"],
                "with_kvcache_mb": r2["peak_mb"],
                "kvcache_mb": r2["kvcache_mb"],
                "saved_pct": saved_pct,
            })

    # Summary
    print(f"\n\n{'='*80}")
    print("VRAM Savings: Prefill-Only (no KV cache) vs Simulated vLLM (with KV cache)")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'No KV (MB)':>12} {'With KV (MB)':>13} {'KV Cache (MB)':>14} {'Saved':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<30} {r['no_kvcache_mb']:>12.0f} {r['with_kvcache_mb']:>13.0f} {r['kvcache_mb']:>14.0f} {r['saved_pct']:>9.1f}%")


if __name__ == "__main__":
    main()
