"""
Benchmark: VRAM savings for embedding/reranker models.

Compares:
  - Our framework (no KV cache) - is_embedding/is_reranker=True
  - Simulated official vLLM (with KV cache) - load same model as generation model

Usage:
    GPU_ID=2 python benchmark_embed_rerank_vram.py
"""
import gc
import os
import random
import time
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU_ID", "2")
os.environ["MASTER_ADDR"] = "127.0.0.1"

from nanovllm import LLM

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub/hf_cache")

MODELS = {
    "qwen3-embedding-0.6b": {
        "path": f"{HF_CACHE}/Qwen3-Embedding-0.6B",
        "prefill_kwargs": {
            "is_embedding": True, "embedding_type": "qwen3",
            "pooling_type": "LAST", "normalize_embeddings": True,
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
        "kvcache_kwargs": {
            "enforce_eager": True, "tensor_parallel_size": 1,
            # Load as plain generation model to force KV cache
        },
    },
    "bge-gemma2": {
        "path": f"{HF_CACHE}/bge-multilingual-gemma2",
        "prefill_kwargs": {
            "is_embedding": True, "embedding_type": "gemma2",
            "pooling_type": "LAST", "normalize_embeddings": True,
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
        "kvcache_kwargs": {
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
    "qwen3-reranker-0.6b": {
        "path": f"{HF_CACHE}/Qwen3-Reranker-0.6B",
        "prefill_kwargs": {
            "is_reranker": True, "reranker_type": "qwen3",
            "is_original_qwen3_reranker": True, "classifier_from_token": ["no", "yes"],
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
        "kvcache_kwargs": {
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
    "jina-reranker-v3": {
        "path": f"{HF_CACHE}/jina-reranker-v3",
        "prefill_kwargs": {
            "is_reranker": True, "reranker_type": "jina_v3",
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
        "kvcache_kwargs": {
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
    "bge-reranker-gemma": {
        "path": f"{HF_CACHE}/bge-reranker-v2-gemma",
        "prefill_kwargs": {
            "is_reranker": True, "reranker_type": "gemma",
            "is_original_qwen3_reranker": True, "classifier_from_token": ["Yes"],
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
        "kvcache_kwargs": {
            "enforce_eager": True, "tensor_parallel_size": 1,
        },
    },
}


def cleanup():
    import torch.distributed as dist
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    os.environ["MASTER_PORT"] = str(random.randint(30000, 60000))
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def main():
    results = []

    for name, cfg in MODELS.items():
        if not os.path.exists(cfg["path"]):
            print(f"SKIP: {name} (not found at {cfg['path']})")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")

        # 1. No KV cache (our framework)
        print(f"  [1/2] Prefill-Only (no KV cache) ...")
        cleanup()
        peak_no_kvcache = None
        try:
            llm = LLM(cfg["path"], **cfg["prefill_kwargs"])
            torch.cuda.synchronize()
            peak_no_kvcache = torch.cuda.max_memory_allocated() / 1024**2
            print(f"    Peak VRAM: {peak_no_kvcache:.0f} MB")

            # Also get KV cache info
            mr = llm.model_runner
            kvcache_status = "allocated" if mr.kv_cache is not None else "None"
            num_blocks = mr.config.num_kvcache_blocks
            print(f"    KV cache: {kvcache_status}, num_blocks={num_blocks}")
            llm.exit()
        except Exception as e:
            print(f"    FAILED: {e}")
        finally:
            cleanup()

        # 2. With KV cache (simulating official vLLM - load as generation model)
        print(f"  [2/2] Simulated vLLM (with KV cache) ...")
        peak_with_kvcache = None
        kvcache_mb = None
        try:
            llm = LLM(cfg["path"], **cfg["kvcache_kwargs"])
            torch.cuda.synchronize()
            peak_with_kvcache = torch.cuda.max_memory_allocated() / 1024**2
            print(f"    Peak VRAM: {peak_with_kvcache:.0f} MB")

            mr = llm.model_runner
            kvcache_status = "allocated" if mr.kv_cache is not None else "None"
            num_blocks = mr.config.num_kvcache_blocks
            print(f"    KV cache: {kvcache_status}, num_blocks={num_blocks}")
            if mr.kv_cache is not None:
                kvcache_mb = mr.kv_cache.numel() * mr.kv_cache.element_size() / 1024**2
                print(f"    KV cache size: {kvcache_mb:.0f} MB")
            llm.exit()
        except Exception as e:
            print(f"    FAILED: {e}")
        finally:
            cleanup()

        if peak_no_kvcache and peak_with_kvcache:
            saved_pct = (1 - peak_no_kvcache / peak_with_kvcache) * 100
            ratio = peak_with_kvcache / peak_no_kvcache
            results.append({
                "name": name,
                "no_kvcache_mb": peak_no_kvcache,
                "with_kvcache_mb": peak_with_kvcache,
                "kvcache_mb": kvcache_mb or 0,
                "saved_pct": saved_pct,
                "ratio": ratio,
            })

    # Print summary
    print(f"\n\n{'='*80}")
    print("VRAM Savings: Prefill-Only (no KV cache) vs Simulated vLLM (with KV cache)")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'No KV (MB)':>12} {'With KV (MB)':>13} {'KV Cache (MB)':>14} {'Saved':>10} {'Ratio':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<30} {r['no_kvcache_mb']:>12.0f} {r['with_kvcache_mb']:>13.0f} {r['kvcache_mb']:>14.0f} {r['saved_pct']:>9.1f}% {r['ratio']:>7.2f}x")

    # Write to report
    print(f"\n\nKey Insight:")
    print(f"Official vLLM allocates KV cache even for embedding/reranker models,")
    print(f"even though it's never used. Our framework skips this allocation entirely,")
    print(f"saving significant VRAM that can be used for larger batch sizes.")


if __name__ == "__main__":
    main()
