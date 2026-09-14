"""Shared fixtures and helpers for the nano-vllm-prefillonly test suite."""
import os

import pytest

MODEL_ROOT = os.environ.get("NANOVLLM_TEST_MODEL_ROOT")

MODELS = {
    "qwen3": "Qwen3-0.6B",
    "qwen3_embedding": "Qwen3-Embedding-0.6B",
    "qwen3_reranker": "Qwen3-Reranker-0.6B",
    "qwen3_vl": "Qwen3-VL-2B-Instruct",
    "qwen2_5_vl": "Qwen2.5-VL-3B-Instruct",
    "qwen3_vl_embedding": "Qwen3-VL-Embedding-2B",
    "qwen3_vl_reranker": "Qwen3-VL-Reranker-2B",
    "gemma2_embedding": "bge-multilingual-gemma2",
    "gemma_reranker": "bge-reranker-v2-gemma",
    "qwen3_5": "Qwen3.5-0.8B",
}


def model_path(key: str) -> str:
    """Resolve a logical model key to an on-disk path, skipping if absent."""
    if not MODEL_ROOT:
        pytest.skip(
            "NANOVLLM_TEST_MODEL_ROOT is not set; point it at a directory "
            "containing the HuggingFace checkpoints listed in tests/conftest.py "
            "(e.g. Qwen3-0.6B, Qwen3-VL-Embedding-2B, ...)"
        )
    name = MODELS.get(key, key)
    path = os.path.join(MODEL_ROOT, name)
    if not os.path.isdir(path):
        pytest.skip(f"model not available: {path}")
    return path


def requires_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@pytest.fixture(autouse=True)
def _cuda_cleanup():
    """Release CUDA memory between tests so large models can run back to back."""
    yield
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
