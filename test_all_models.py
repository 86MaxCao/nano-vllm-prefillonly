"""Unified test script for all supported model types.

Tests each model category: text generation, VL generation, embedding, reranking.

Usage:
    micromamba activate main
    python test_all_models.py [--category CATEGORY] [--model MODEL_NAME]

Categories: generation, vl_generation, embedding, vl_embedding, reranker, vl_reranker, all
"""

import argparse
import os
import sys
import time
import traceback

import torch

MODEL_CACHE = os.path.expanduser("~/.cache/huggingface/hub/hf_cache/")

# ── Model configurations ──────────────────────────────────────────────────────

MODELS = {
    # Text generation
    "generation": {
        "Qwen3-4B": {
            "path": os.path.join(MODEL_CACHE, "Qwen3-4B"),
            "kwargs": {},
        },
    },
    # VL generation (multimodal)
    "vl_generation": {
        "Qwen3-VL-2B": {
            "path": os.path.join(MODEL_CACHE, "Qwen3-VL-2B-Instruct"),
            "kwargs": {"is_multimodal": True, "multimodal_model_type": "qwen3_vl"},
        },
        "Qwen2-VL-2B": {
            "path": os.path.join(MODEL_CACHE, "Qwen2-VL-2B-Instruct"),
            "kwargs": {"is_multimodal": True, "multimodal_model_type": "qwen2_vl"},
        },
        "Qwen2.5-VL-3B": {
            "path": os.path.join(MODEL_CACHE, "Qwen2.5-VL-3B-Instruct"),
            "kwargs": {"is_multimodal": True, "multimodal_model_type": "qwen2_5_vl"},
        },
    },
    # Text embedding
    "embedding": {
        "Qwen3-Embedding": {
            "path": os.path.join(MODEL_CACHE, "Qwen3-Embedding-0.6B"),
            "kwargs": {"is_embedding": True, "embedding_type": "qwen3"},
        },
        "BGE-Gemma2": {
            "path": os.path.join(MODEL_CACHE, "bge-multilingual-gemma2"),
            "kwargs": {"is_embedding": True, "embedding_type": "gemma2"},
        },
    },
    # VL embedding
    "vl_embedding": {
        "Qwen3-VL-Embedding": {
            "path": os.path.join(MODEL_CACHE, "Qwen3-VL-Embedding-2B"),
            "kwargs": {"is_embedding": True, "embedding_type": "qwen3_vl"},
        },
        "Jina-V4": {
            "path": os.path.join(MODEL_CACHE, "jina-embeddings-v4"),
            "kwargs": {
                "is_embedding": True,
                "embedding_type": "jina_v4",
                "trust_remote_code": True,
            },
        },
    },
    # Text reranker
    "reranker": {
        "Qwen3-Reranker": {
            "path": os.path.join(MODEL_CACHE, "Qwen3-Reranker-0.6B"),
            "kwargs": {
                "is_reranker": True,
                "reranker_type": "qwen3",
                "is_original_qwen3_reranker": True,
                "classifier_from_token": ["no", "yes"],
            },
        },
        "Jina-Reranker-V3": {
            "path": os.path.join(MODEL_CACHE, "jina-reranker-v3"),
            "kwargs": {
                "is_reranker": True,
                "reranker_type": "jina_v3",
                "trust_remote_code": True,
            },
        },
    },
    # VL reranker
    "vl_reranker": {
        "Qwen3-VL-Reranker": {
            "path": os.path.join(MODEL_CACHE, "Qwen3-VL-Reranker-2B"),
            "kwargs": {
                "is_reranker": True,
                "reranker_type": "qwen3_vl",
                "is_original_qwen3_reranker": True,
                "classifier_from_token": ["no", "yes"],
                "trust_remote_code": True,
            },
        },
        "Jina-Reranker-M0": {
            "path": os.path.join(MODEL_CACHE, "jina-reranker-m0"),
            "kwargs": {
                "is_reranker": True,
                "reranker_type": "jina_m0",
                "trust_remote_code": True,
            },
        },
    },
}


def check_model_path(path: str) -> bool:
    """Check if model path exists."""
    return os.path.isdir(path)


def test_generation(model_name: str, config: dict):
    """Test text generation model."""
    from nanovllm import LLM, SamplingParams

    path = config["path"]
    kwargs = config["kwargs"]

    print(f"  Loading {model_name} from {path}...")
    llm = LLM(path, enforce_eager=True, **kwargs)

    prompts = [
        "The capital of France is",
        "The meaning of life is",
    ]
    sp = SamplingParams(max_tokens=32, temperature=0.0)

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    for i, out in enumerate(outputs):
        text = out["text"][:80]
        print(f"    Prompt {i}: {prompts[i][:30]}... -> {text}...")

    print(f"    Time: {elapsed:.2f}s")
    del llm
    torch.cuda.empty_cache()
    return True


def test_vl_generation(model_name: str, config: dict):
    """Test VL generation model."""
    from nanovllm import LLM, SamplingParams
    from transformers import AutoProcessor

    path = config["path"]
    kwargs = config["kwargs"]

    print(f"  Loading {model_name} from {path}...")
    llm = LLM(path, enforce_eager=True, **kwargs)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    # Use a simple text-only request for basic test
    requests = [
        {"text": "Describe the capital of France in one sentence.", "images": None},
    ]
    sp = SamplingParams(max_tokens=32, temperature=0.0)

    try:
        t0 = time.time()
        outputs = llm.generate_multimodal(requests, sp, processor)
        elapsed = time.time() - t0

        for i, out in enumerate(outputs):
            text = out["text"][:80]
            print(f"    Output {i}: {text}...")

        print(f"    Time: {elapsed:.2f}s")
    except Exception as e:
        print(f"    ERROR: {e}")
        traceback.print_exc()
        del llm
        torch.cuda.empty_cache()
        return False

    del llm
    torch.cuda.empty_cache()
    return True


def test_embedding(model_name: str, config: dict):
    """Test embedding model."""
    from nanovllm import LLM

    path = config["path"]
    kwargs = config["kwargs"]

    print(f"  Loading {model_name} from {path}...")
    llm = LLM(path, enforce_eager=True, **kwargs)

    texts = [
        "What is the capital of France?",
        "Paris is the capital of France.",
        "The capital of Germany is Berlin.",
    ]

    t0 = time.time()
    embeddings = llm.embed_batch(texts)
    elapsed = time.time() - t0

    print(f"    Embeddings shape: {embeddings.shape}")
    # Compute similarity
    sim_01 = torch.nn.functional.cosine_similarity(
        embeddings[0:1], embeddings[1:2]
    ).item()
    sim_02 = torch.nn.functional.cosine_similarity(
        embeddings[0:1], embeddings[2:3]
    ).item()
    print(f"    Sim(text0, text1): {sim_01:.4f} (should be high)")
    print(f"    Sim(text0, text2): {sim_02:.4f} (should be lower)")
    print(f"    Time: {elapsed:.2f}s")

    del llm
    torch.cuda.empty_cache()
    return True


def test_vl_embedding(model_name: str, config: dict):
    """Test VL embedding model."""
    from nanovllm import LLM

    path = config["path"]
    kwargs = config["kwargs"]

    print(f"  Loading {model_name} from {path}...")
    llm = LLM(path, enforce_eager=True, **kwargs)

    texts = [
        "What is the capital of France?",
        "Paris is the capital of France.",
    ]

    try:
        t0 = time.time()
        embeddings = llm.embed_batch(texts)
        elapsed = time.time() - t0

        print(f"    Embeddings shape: {embeddings.shape}")
        sim_01 = torch.nn.functional.cosine_similarity(
            embeddings[0:1], embeddings[1:2]
        ).item()
        print(f"    Sim(text0, text1): {sim_01:.4f}")
        print(f"    Time: {elapsed:.2f}s")
    except Exception as e:
        print(f"    ERROR: {e}")
        traceback.print_exc()
        del llm
        torch.cuda.empty_cache()
        return False

    del llm
    torch.cuda.empty_cache()
    return True


def test_reranker(model_name: str, config: dict):
    """Test reranker model."""
    from nanovllm import LLM

    path = config["path"]
    kwargs = config["kwargs"]

    print(f"  Loading {model_name} from {path}...")
    llm = LLM(path, enforce_eager=True, **kwargs)

    pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "Berlin is the capital of Germany."),
    ]

    try:
        t0 = time.time()
        result = llm.rerank_batch(pairs)
        elapsed = time.time() - t0

        if isinstance(result, tuple):
            scores = result[0]
        else:
            scores = result
        print(f"    Scores shape: {scores.shape if hasattr(scores, 'shape') else 'N/A'}")
        print(f"    Score(relevant): {scores[0].item():.4f} (should be high)")
        print(f"    Score(irrelevant): {scores[1].item():.4f} (should be lower)")
        print(f"    Time: {elapsed:.2f}s")
    except Exception as e:
        print(f"    ERROR: {e}")
        traceback.print_exc()
        del llm
        torch.cuda.empty_cache()
        return False

    del llm
    torch.cuda.empty_cache()
    return True


TEST_FUNCS = {
    "generation": test_generation,
    "vl_generation": test_vl_generation,
    "embedding": test_embedding,
    "vl_embedding": test_vl_embedding,
    "reranker": test_reranker,
    "vl_reranker": test_reranker,
}


def main():
    parser = argparse.ArgumentParser(description="Test all supported models")
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=list(MODELS.keys()) + ["all"],
        help="Model category to test",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name to test",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip models whose path doesn't exist",
    )
    args = parser.parse_args()

    categories = list(MODELS.keys()) if args.category == "all" else [args.category]

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for cat in categories:
        print(f"\n{'='*60}")
        print(f"  Category: {cat}")
        print(f"{'='*60}")

        for model_name, config in MODELS[cat].items():
            if args.model and model_name != args.model:
                continue

            total += 1

            if not check_model_path(config["path"]):
                if args.skip_missing:
                    print(f"  SKIP {model_name}: path not found ({config['path']})")
                    skipped += 1
                    continue
                else:
                    print(f"  SKIP {model_name}: path not found ({config['path']})")
                    print(f"    Use --skip-missing to suppress")
                    skipped += 1
                    continue

            print(f"\n  Testing: {model_name}")
            try:
                ok = TEST_FUNCS[cat](model_name, config)
                if ok:
                    passed += 1
                    print(f"  PASS {model_name}")
                else:
                    failed += 1
                    print(f"  FAIL {model_name}")
            except Exception as e:
                failed += 1
                print(f"  FAIL {model_name}: {e}")
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped / {total} total")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
