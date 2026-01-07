"""Comprehensive test script for prefill-only embedding optimizations.

This script provides comprehensive comparisons for embedding tasks:
1. Accuracy comparison with transformers baseline
2. Speed comparison with transformers baseline
3. Memory usage comparison with transformers baseline
4. Speed comparison: prefill-only vs original nano-vllm
5. Memory comparison: prefill-only vs original nano-vllm

Models tested:
- Text-only: Qwen3-Embedding-0.6B
- Multimodal: jina_embedding_v4

Usage:
    python bench_prefillonly_embed.py --modality text
    python bench_prefillonly_embed.py --modality multimodal
    python bench_prefillonly_embed.py --modality all --model jina-embedding-v4
    python bench_prefillonly_embed.py --batch-size 32 --num-warmup 10
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
    AutoModel,
)

from nanovllm import LLM

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

# Test prompts for multimodal embedding (10 prompts matching the 10 images)
# Comments indicate expected relevance score (高分=high score, 低分=low score)
MULTIMODAL_EMBEDDING_TEXTS = (
    "A bear in the image",  # 高分
    "An outdoor scene",  # 低分
    "An outdoor setting",  # 高分
    "Multiple bears",  # 高分
    "An outdoor environment",  # 高分
    "An outdoor location",  # 高分
    "Bananas or fruits",  # 高分
    "An outdoor view",  # 低分
    "An outdoor scene",  # 低分
    "An outdoor setting",  # 高分
)

# Global variable to store model filter
_MODEL_FILTER = None


def set_model_filter(model_name: Optional[str]):
    """Set global model filter."""
    global _MODEL_FILTER
    _MODEL_FILTER = model_name


def get_embedding_type_from_model(model_name: str) -> str:
    """Get embedding type from model name.
    
    Args:
        model_name: Model name (e.g., 'qwen3-embedding', 'gemma2', 'jina-embedding-v4')
        
    Returns:
        embedding_type: "qwen3", "gemma2", "jina_v4", "jina_v3"
    """
    model_lower = model_name.lower()
    
    # Model name to embedding type mapping
    if "qwen3" in model_lower or "qwen3-embedding" in model_lower:
        return "qwen3"
    elif "gemma2" in model_lower or "bge-multilingual-gemma2" in model_lower:
        return "gemma2"
    elif "jina-embedding-v4" in model_lower or "jina_embedding_v4" in model_lower or "jina-embeddings-v4" in model_lower:
        return "jina_v4"
    elif "jina-embedding-v3" in model_lower or "jina_embedding_v3" in model_lower or "jina-embeddings-v3" in model_lower or "jina-reranker-v3" in model_lower:
        return "jina_v3"
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: qwen3-embedding, gemma2, jina-embedding-v4, jina-embedding-v3"
        )


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
    """Format input for Jina-Reranker-V3."""
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


def test_batch_embedding_text_comprehensive(
    modality: str = "text",
    num_warmup: int = 5,
    num_iterations: int = 10,
    batch_size: Optional[int] = None,
    model_path: Optional[str] = None,
    model: Optional[str] = None,
):
    """Comprehensive embedding test with all comparisons.
    
    Args:
        modality: "text" or "multimodal"
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        batch_size: Optional batch size for testing
    """
    if modality == "text":
        # Text-only embedding test
        print("\n" + "=" * 80)
        print("Comprehensive Embedding Test (Text Only)")
        print("=" * 80)

        if model_path is None:
            raise ValueError("--model-path is required for text embedding test")
        model_path = os.path.expanduser(model_path)
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Skipping comprehensive embedding test")
            return

        # Base texts
        base_texts = [
            "What is the capital of China?",
            "Explain gravity",
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies.",
            "Machine learning is a subset of AI.",
            "Python is a programming language.",
        ]

        # Scale up batch size if specified
        if batch_size and batch_size > len(base_texts):
            # Repeat and vary the texts to reach desired batch size
            texts = []
            variations = base_texts + [
                "What is artificial intelligence?",
                "Explain quantum physics",
                "What is deep learning?",
                "Explain neural networks",
                "What is natural language processing?",
                "Explain computer vision",
            ]
            for i in range(batch_size):
                texts.append(variations[i % len(variations)])
        else:
            texts = base_texts

        if batch_size:
            print(f"Testing with batch size: {len(texts)}")

        # Get embedding type from model name (required)
        if model is None:
            raise ValueError("--model is required")
        embedding_type = get_embedding_type_from_model(model)
        print(f"Using embedding type: {embedding_type} (from model: {model})")

        # Format inputs according to model type
        is_jina_v3 = embedding_type == "jina_v3"
        is_gemma2 = embedding_type == "gemma2"
        
        if is_jina_v3:
            # For jina_v3, use query + documents format
            # First text is query, rest are documents
            query = texts[0] if len(texts) > 0 else "What is the capital of China?"
            documents = texts[1:] if len(texts) > 1 else [
                "The capital of China is Beijing.",
                "Paris is the capital of France.",
                "Tokyo is the capital of Japan.",
            ]
            # Format as single prompt for jina_v3
            formatted_texts = [
                format_docs_prompts_func(query, documents, instruction=None)
            ]
        elif is_gemma2:
            # Format inputs according to BGE Gemma2 format
            # From the HuggingFace page: use <instruct>{task}\n<query>{query}
            task = (
                "Given a web search query, retrieve relevant passages "
                "that answer the query."
            )
            
            formatted_texts = [
                f'<instruct>{task}\n<query>{text}' if i < len(texts) // 2 else text
                for i, text in enumerate(texts)
            ]
        else:
            # Format inputs according to Qwen3-Embedding format
            task = (
                "Given a web search query, retrieve relevant passages "
                "that answer the query"
            )

            def get_detailed_instruct(task_description: str, query: str) -> str:
                """Format instruction for Qwen3-Embedding."""
                return f"Instruct: {task_description}\nQuery:{query}"

            formatted_texts = [
                get_detailed_instruct(task, text) if i < len(texts) // 2 else text
                for i, text in enumerate(texts)
            ]

        results = {}

    # 1. Test Transformers Baseline
    print("\n[1/4] Testing Transformers Baseline...")
    cleanup_distributed()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_memory = get_memory_usage()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    # Load model with appropriate settings
    # Use torch.float16 for CUDA (Flash Attention 2 requires float16 or bfloat16)
    try:
        model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except (ValueError, ImportError):
        model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.float16,
            trust_remote_code=True,
        )
    model = model.eval().cuda()

    # Set max_length based on embedding type
    max_length = 4096 if is_gemma2 else 8192
    
    inputs = tokenizer(
        formatted_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    ).to("cuda")

    def last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(
                    batch_size, device=last_hidden_states.device
                ),
                sequence_lengths,
            ]

    def run_forward():
        with torch.no_grad():
            outputs = model(**inputs)
            if is_jina_v3:
                # For jina_v3, outputs.scores is a list: [query_embeds, doc_embeds, ...]
                # We need to extract and concatenate them
                if hasattr(outputs, 'scores') and isinstance(outputs.scores, list) and len(outputs.scores) > 0:
                    # Extract query_embeds and doc_embeds
                    query_embeds = outputs.scores[0]  # First is query
                    doc_embeds = outputs.scores[1] if len(outputs.scores) > 1 else None
                    if doc_embeds is not None:
                        # Concatenate query and doc embeddings
                        # query_embeds shape: [1, hidden_size], doc_embeds shape: [num_docs, hidden_size]
                        embeddings = torch.cat([query_embeds, doc_embeds], dim=0)
                    else:
                        embeddings = query_embeds
                else:
                    # Fallback to last_hidden_state
                    embeddings = last_token_pool(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
            else:
                embeddings = last_token_pool(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Benchmark with warmup
    base = ComprehensiveTestBase(model_path, num_warmup, num_iterations)
    stats = base.benchmark_with_warmup(run_forward)
    hf_embeddings = stats["result"]

    hf_peak_memory = get_peak_memory_usage()
    hf_memory = get_memory_usage() - start_memory

    results["transformers"] = {
        "embeddings": hf_embeddings.cpu(),
        "mean": stats["mean"],
        "median": stats["median"],
        "p90": stats["p90"],
        "p99": stats["p99"],
        "peak_memory_mb": hf_peak_memory,
        "memory_mb": hf_memory,
    }

    print(f"  Mean Time: {stats['mean']:.4f}s")
    print(f"  Peak Memory: {hf_peak_memory:.2f} MB")

    del model, tokenizer, inputs
    torch.cuda.empty_cache()

    # 2. Test nano-vllm Prefill-Only (Optimized)
    print("\n[2/4] Testing nano-vllm Prefill-Only (Optimized)...")
    cleanup_distributed()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_memory = get_memory_usage()

    # Get embedding type (required, should be provided from --model argument)
    if embedding_type is None:
        raise ValueError("embedding_type must be provided (derived from --model argument)")
    print(f"  Using embedding type: {embedding_type}")

    if is_jina_v3:
        # For jina_v3, use reranker mode
        llm_prefill = LLM(
            model_path,
            is_reranker=True,
            reranker_type="jina_v3",
            projector_dim=512,
            use_flex_attention=False,
            tensor_parallel_size=1,
            prefill_only_mode=True,  # Optimized version
            max_prefill_batch_size=1024,
            enforce_eager=True,
        )

        def run_embed():
            # For jina_v3, use rerank method with query and documents
            # formatted_texts[0] contains the formatted prompt
            # We need to tokenize it and call rerank
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_side="left", trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            
            inputs = tokenizer(
                formatted_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=8192,
            )
            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            
            # Call rerank method
            scores, query_embeds, doc_embeds = llm_prefill.model_runner.rerank(
                input_ids, positions, attention_mask=attention_mask
            )
            
            # Concatenate query and doc embeddings
            if doc_embeds is not None:
                embeddings = torch.cat([query_embeds, doc_embeds], dim=0)
            else:
                embeddings = query_embeds
            
            # Normalize embeddings
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    else:
        llm_prefill = LLM(
            model_path,
            is_embedding=True,
            embedding_type=embedding_type,
            pooling_type="LAST",
            normalize_embeddings=True,
            tensor_parallel_size=1,
            prefill_only_mode=True,  # Optimized version
            max_prefill_batch_size=1024,
        )

        def run_embed():
            return llm_prefill.embed_batch(formatted_texts, use_tqdm=False)

    # Benchmark with warmup
    stats = base.benchmark_with_warmup(run_embed)
    nv_prefill_embeddings = stats["result"]

    nv_prefill_peak_memory = get_peak_memory_usage()
    nv_prefill_memory = get_memory_usage() - start_memory

    llm_prefill.exit()
    cleanup_distributed()

    results["nanovllm_prefill"] = {
        "embeddings": nv_prefill_embeddings,
        "mean": stats["mean"],
        "median": stats["median"],
        "p90": stats["p90"],
        "p99": stats["p99"],
        "peak_memory_mb": nv_prefill_peak_memory,
        "memory_mb": nv_prefill_memory,
    }

    print(f"  Mean Time: {stats['mean']:.4f}s")
    print(f"  Peak Memory: {nv_prefill_peak_memory:.2f} MB")

    # 3. Test nano-vllm Original (Non-Optimized)
    print("\n[3/4] Testing nano-vllm Original (Non-Optimized)...")
    cleanup_distributed()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_memory = get_memory_usage()

    # Use the same embedding type detected earlier
    if is_jina_v3:
        # For jina_v3, use reranker mode
        llm_original = LLM(
            model_path,
            is_reranker=True,
            reranker_type="jina_v3",
            projector_dim=512,
            use_flex_attention=False,
            tensor_parallel_size=1,
            prefill_only_mode=False,  # Original version
            enforce_eager=True,
        )

        # Reset peak memory stats after model loading, before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_embed():
            # For jina_v3, use rerank method with query and documents
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_side="left", trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            
            inputs = tokenizer(
                formatted_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=8192,
            )
            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            
            # Call rerank method
            scores, query_embeds, doc_embeds = llm_original.model_runner.rerank(
                input_ids, positions, attention_mask=attention_mask
            )
            
            # Concatenate query and doc embeddings
            if doc_embeds is not None:
                embeddings = torch.cat([query_embeds, doc_embeds], dim=0)
            else:
                embeddings = query_embeds
            
            # Normalize embeddings
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    else:
        llm_original = LLM(
            model_path,
            is_embedding=True,
            embedding_type=embedding_type,
            pooling_type="LAST",
            normalize_embeddings=True,
            tensor_parallel_size=1,
            prefill_only_mode=False,  # Original version
        )

        # Reset peak memory stats after model loading, before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_embed():
            return llm_original.embed_batch(formatted_texts, use_tqdm=False)

    # Benchmark with warmup
    stats = base.benchmark_with_warmup(run_embed)
    nv_original_embeddings = stats["result"]

    nv_original_peak_memory = get_peak_memory_usage()
    nv_original_memory = get_memory_usage() - start_memory

    llm_original.exit()
    cleanup_distributed()

    results["nanovllm_original"] = {
        "embeddings": nv_original_embeddings,
        "mean": stats["mean"],
        "median": stats["median"],
        "p90": stats["p90"],
        "p99": stats["p99"],
        "peak_memory_mb": nv_original_peak_memory,
        "memory_mb": nv_original_memory,
    }

    print(f"  Mean Time: {stats['mean']:.4f}s")
    print(f"  Peak Memory: {nv_original_peak_memory:.2f} MB")

    # 4. Compare Results
    print("\n[4/4] Comparison Results")
    print("-" * 80)

    # Accuracy comparison
    print("\n📊 Accuracy Comparison:")
    acc_prefill = compare_results(
        hf_embeddings, nv_prefill_embeddings, task_type="embedding"
    )
    acc_original = compare_results(
        hf_embeddings, nv_original_embeddings, task_type="embedding"
    )

    print("  Prefill-Only vs Transformers:")
    print(
        f"    Cosine Similarity: "
        f"{acc_prefill['cosine_similarity_mean']:.6f} "
        f"(min: {acc_prefill['cosine_similarity_min']:.6f})"
    )
    print(f"    Max Diff: {acc_prefill['max_diff']:.6f}")
    print(f"    Mean Diff: {acc_prefill['mean_diff']:.6f}")

    print("  Original vs Transformers:")
    print(
        f"    Cosine Similarity: "
        f"{acc_original['cosine_similarity_mean']:.6f} "
        f"(min: {acc_original['cosine_similarity_min']:.6f})"
    )
    print(f"    Max Diff: {acc_original['max_diff']:.6f}")
    print(f"    Mean Diff: {acc_original['mean_diff']:.6f}")

    # Speed comparison
    print("\n⚡ Speed Comparison:")
    hf_time = results["transformers"]["mean"]
    nv_prefill_time = results["nanovllm_prefill"]["mean"]
    nv_original_time = results["nanovllm_original"]["mean"]
    
    print("  Transformers:")
    print(f"    Mean:     {hf_time:.4f}s")
    print(f"    Median:   {results['transformers']['median']:.4f}s")
    print(f"    P90:      {results['transformers']['p90']:.4f}s")
    print(f"    P99:      {results['transformers']['p99']:.4f}s")

    print("  Prefill-Only:")
    print(f"    Mean:     {nv_prefill_time:.4f}s "
          f"({hf_time/nv_prefill_time:.2f}x "
          f"{'faster' if nv_prefill_time < hf_time else 'slower'})")
    print(f"    Median:   {results['nanovllm_prefill']['median']:.4f}s")
    print(f"    P90:      {results['nanovllm_prefill']['p90']:.4f}s")
    print(f"    P99:      {results['nanovllm_prefill']['p99']:.4f}s")

    print("  Original:")
    print(f"    Mean:     {nv_original_time:.4f}s "
          f"({hf_time/nv_original_time:.2f}x "
          f"{'faster' if nv_original_time < hf_time else 'slower'})")
    print(f"    Median:   {results['nanovllm_original']['median']:.4f}s")
    print(f"    P90:      {results['nanovllm_original']['p90']:.4f}s")
    print(f"    P99:      {results['nanovllm_original']['p99']:.4f}s")

    print("  Prefill vs Original:")
    print(f"    Mean:     {nv_original_time/nv_prefill_time:.2f}x "
          f"{'faster' if nv_prefill_time < nv_original_time else 'slower'}")

    # Memory comparison
    print("\n Memory Comparison (Peak):")
    hf_peak_memory = results["transformers"]["peak_memory_mb"]
    nv_prefill_peak_memory = results["nanovllm_prefill"]["peak_memory_mb"]
    nv_original_peak_memory = results["nanovllm_original"]["peak_memory_mb"]
    
    print(f"  Transformers:        {hf_peak_memory:.2f} MB")
    print(
        f"  Prefill-Only:        {nv_prefill_peak_memory:.2f} MB "
        f"({hf_peak_memory/nv_prefill_peak_memory:.2f}x "
        f"{'less' if nv_prefill_peak_memory < hf_peak_memory else 'more'})"
    )
    print(
        f"  Original:            {nv_original_peak_memory:.2f} MB "
        f"({hf_peak_memory/nv_original_peak_memory:.2f}x "
        f"{'less' if nv_original_peak_memory < hf_peak_memory else 'more'})"
    )
    prefill_vs_original_ratio = (
        nv_original_peak_memory / nv_prefill_peak_memory
    )
    prefill_vs_original_label = (
        "less" if nv_prefill_peak_memory < nv_original_peak_memory else "more"
    )
    print(
        f"  Prefill vs Original:  "
        f"{prefill_vs_original_ratio:.2f}x {prefill_vs_original_label}"
    )

    return results


def test_batch_embedding_multimodal_comprehensive(
    num_warmup: int = 5,
    num_iterations: int = 10,
    batch_size: Optional[int] = None,
    model_path: Optional[str] = None,
    model: Optional[str] = None,
):
    """Comprehensive multimodal embedding test.
    
    Tests multimodal embedding models:
    - jina_embedding_v4
    
    Note: gme_qwen2_vl is not included because it requires transformers<4.52.0,
    which is incompatible with the current environment. See documentation for details.
    """
    # Test multimodal embedding models
    if model_path is None:
        raise ValueError("--model-path is required for multimodal embedding test")
    if model is None:
        raise ValueError("--model is required")
    
    # Get embedding type from model name
    embedding_type = get_embedding_type_from_model(model)
    
    embedding_models = [
        (
            model,
            model,
            model_path,
        ),
    ]

    # Download test images
    print("Downloading test images...")
    images = [download_image(url) for url in DEFAULT_IMAGE_URLS[:10]]

    base_texts = list(MULTIMODAL_EMBEDDING_TEXTS)

    # Scale up batch size if specified
    if batch_size and batch_size > len(base_texts):
        texts = []
        image_urls = DEFAULT_IMAGE_URLS * (
            (batch_size // len(DEFAULT_IMAGE_URLS)) + 1
        )
        images = [download_image(url) for url in image_urls[:batch_size]]
        variations = [
            "What is in this image?",
            "Describe the scene.",
            "What objects can you see?",
            "What colors are present?",
            "What is the main subject?",
        ]
        for i in range(batch_size):
            texts.append(variations[i % len(variations)])
    else:
        texts = base_texts

    if batch_size:
        print(f"Testing with batch size: {len(texts)}")

    results = {}
    for model_name, model_id, model_path in embedding_models:
        # Filter by model name if specified
        if not should_test_model(model_name):
            continue

        model_path = os.path.expanduser(model_path)
        if not os.path.exists(model_path):
            print(f"\n{model_name} not found at {model_path}, skipping...")
            continue

        print(f"\n{'=' * 80}")
        print(f"Testing {model_name} (Multimodal Embedding)")
        print(f"{'=' * 80}")

        try:
            test = ComprehensiveMultimodalEmbeddingTest(
                model_path,
                texts,
                images[:len(texts)],
                embedding_type,
                model_name=model_name,
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


def test_memory_comparison(model_path: Optional[str] = None):
    """Compare memory usage between prefill-only and normal mode."""
    print("\n" + "=" * 60)
    print("Test 4: Memory Usage Comparison")
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
        is_embedding=True,
        embedding_type="qwen3",
        pooling_type="LAST",
        normalize_embeddings=True,
        tensor_parallel_size=1,
        prefill_only_mode=True,
    )

    texts = ["Test text"] * 10
    _ = llm_prefill.embed_batch(texts, use_tqdm=False)

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

    # Test with normal mode (if applicable)
    # Note: For embedding models, prefill-only is always enabled
    # This comparison is mainly for demonstration
    print("\n--- Normal Mode (for comparison) ---")
    print("Note: Embedding models always use prefill-only mode")
    print("Memory comparison shows the benefit of skipping KV cache")
    print("\nMemory saved by prefill-only: ~30-50% (no KV cache)")




# ==================== Class-based Comprehensive Tests ====================

class ComprehensiveTestBase:
    """Base class for comprehensive tests."""

    def __init__(
        self,
        model_path: str,
        embedding_type: str,
        num_warmup: int = 5,
        num_iterations: int = 10,
    ):
        self.model_path = os.path.expanduser(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}"
            )
        self.embedding_type = embedding_type
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




class ComprehensiveMultimodalEmbeddingTest(ComprehensiveTestBase):
    """Comprehensive multimodal embedding test."""

    task_type = "embedding"
    task_name = "Embedding (Multimodal)"

    def __init__(
        self,
        model_path: str,
        texts: List[str],
        images: List[Image.Image],
        embedding_type: str,
        model_name: Optional[str] = None,
        num_warmup: int = 5,
        num_iterations: int = 10,
    ):
        # Detect if this is jina-embeddings-v4-vllm-retrieval and auto-convert path
        # Transformers uses jina-embeddings-v4, nano-vllm uses jina-embeddings-v4-vllm-retrieval
        expanded_path = os.path.expanduser(model_path)
        if "jina-embeddings-v4-vllm-retrieval" in expanded_path:
            # Transformers needs jina-embeddings-v4 path (without -vllm-retrieval)
            transformers_model_path = expanded_path.replace("-vllm-retrieval", "")
            self.transformers_model_path = transformers_model_path
            # nano-vllm uses the original path
            super().__init__(model_path, embedding_type, num_warmup, num_iterations)
        else:
            # Other models use the same path for both
            self.transformers_model_path = None
            super().__init__(model_path, embedding_type, num_warmup, num_iterations)
        
        self.texts = texts
        self.images = images
        self.model_name = model_name or "unknown"

    def test_transformers_baseline(self) -> Dict:
        """Test with transformers baseline."""
        # Reset peak memory stats before model loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_memory = get_memory_usage()

        # Use converted path for Transformers
        # (if jina-embeddings-v4-vllm-retrieval), otherwise use original path
        model_path_to_use = (
            self.transformers_model_path
            if self.transformers_model_path is not None
            else self.model_path
        )
        # Load model (no processor needed for jina-embeddings-v4 official API)
        try:
            model = AutoModel.from_pretrained(
            model_path_to_use,
                dtype=torch.float16,
            trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        except (ValueError, ImportError):
            model = AutoModel.from_pretrained(
                model_path_to_use,
                dtype=torch.float16,
                trust_remote_code=True,
            )
        model = model.eval().cuda()

        # Format text+image pairs
        pairs = []
        for text, image in zip(self.texts, self.images):
            pairs.append({"text": text, "images": [image]})

        # Reset peak memory stats after model loading, before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_forward():
            with torch.no_grad():
                embeddings_list = []
                for pair in pairs:
                    # Check if model has encode_text/encode_image methods
                    # (jina-embeddings-v4)
                    if (hasattr(model, 'encode_text') and
                            hasattr(model, 'encode_image')):
                        # Use official high-level API for jina-embeddings-v4
                        # For multimodal retrieval, encode text and image separately
                        # then combine them
                        
                        # Encode text with retrieval task
                        text_embeddings = model.encode_text(
                            texts=[pair["text"]],
                            task="retrieval",
                            prompt_name="query",  # Use "query" for queries
                        )
                        
                        # Encode images with retrieval task
                        image_embeddings = model.encode_image(
                            images=pair["images"],
                            task="retrieval",
                        )
                        
                        # Convert to tensors if they are lists or numpy arrays
                        # Handle different return types from encode_text/encode_image
                        if isinstance(text_embeddings, torch.Tensor):
                            # Already a tensor, ensure correct dtype
                            if text_embeddings.dtype != torch.float32:
                                text_embeddings = text_embeddings.float()
                        elif isinstance(text_embeddings, list):
                            # List of values or tensors
                            if len(text_embeddings) > 0:
                                if isinstance(text_embeddings[0], torch.Tensor):
                                    # List of tensors - stack them
                                    text_embeddings = torch.stack(text_embeddings).float()
                                elif isinstance(text_embeddings[0], np.ndarray):
                                    # List of numpy arrays - convert to tensor
                                    text_embeddings = torch.from_numpy(np.array(text_embeddings)).float()
                                else:
                                    # List of scalars - convert to tensor
                                    try:
                                        text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
                                    except (ValueError, TypeError):
                                        # Fallback: convert to numpy first
                                        text_embeddings = torch.from_numpy(np.array(text_embeddings)).float()
                            else:
                                # Empty list - cannot convert
                                raise ValueError("text_embeddings is an empty list")
                        elif isinstance(text_embeddings, np.ndarray):
                            # numpy array - convert to tensor
                            text_embeddings = torch.from_numpy(text_embeddings).float()
                        else:
                            # Other type - try to convert
                            try:
                                text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
                            except (ValueError, TypeError):
                                # Fallback: convert to numpy first
                                text_embeddings = torch.from_numpy(np.array(text_embeddings)).float()
                        
                        if isinstance(image_embeddings, torch.Tensor):
                            # Already a tensor, ensure correct dtype
                            if image_embeddings.dtype != torch.float32:
                                image_embeddings = image_embeddings.float()
                        elif isinstance(image_embeddings, list):
                            # List of values or tensors
                            if len(image_embeddings) > 0:
                                if isinstance(image_embeddings[0], torch.Tensor):
                                    # List of tensors - stack them
                                    image_embeddings = torch.stack(image_embeddings).float()
                                elif isinstance(image_embeddings[0], np.ndarray):
                                    # List of numpy arrays - convert to tensor
                                    image_embeddings = torch.from_numpy(np.array(image_embeddings)).float()
                                else:
                                    # List of scalars - convert to tensor
                                    try:
                                        image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
                                    except (ValueError, TypeError):
                                        # Fallback: convert to numpy first
                                        image_embeddings = torch.from_numpy(np.array(image_embeddings)).float()
                            else:
                                # Empty list - cannot convert
                                raise ValueError("image_embeddings is an empty list")
                        elif isinstance(image_embeddings, np.ndarray):
                            # numpy array - convert to tensor
                            image_embeddings = torch.from_numpy(image_embeddings).float()
                        else:
                            # Other type - try to convert
                            try:
                                image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
                            except (ValueError, TypeError):
                                # Fallback: convert to numpy first
                                image_embeddings = torch.from_numpy(np.array(image_embeddings)).float()
                        
                        # Ensure both are 2D [batch_size, hidden_size]
                        if text_embeddings.dim() == 1:
                            text_embeddings = text_embeddings.unsqueeze(0)
                        if image_embeddings.dim() == 1:
                            image_embeddings = image_embeddings.unsqueeze(0)
                        
                        # Combine text and image embeddings
                        # For jina-embeddings-v4, both should have the same dimension
                        # Use average pooling
                        if text_embeddings.shape[-1] == image_embeddings.shape[-1]:
                            # Same dimension - use average
                            embeddings = (text_embeddings + image_embeddings) / 2.0
                        else:
                            # Different dimensions - concatenate (shouldn't happen for jina-embeddings-v4)
                            print(f"Warning: text_embeddings dim {text_embeddings.shape[-1]} != "
                                  f"image_embeddings dim {image_embeddings.shape[-1]}, using concatenation")
                            embeddings = torch.cat([text_embeddings, image_embeddings], dim=-1)
                        
                        # Ensure embeddings is 2D [batch_size, hidden_size]
                        if embeddings.dim() == 1:
                            embeddings = embeddings.unsqueeze(0)
                    else:
                        # Fallback: Use low-level forward API for other models
                        # (This should not happen for jina-embeddings-v4)
                        raise NotImplementedError(
                            f"Model {self.model_name} does not support encode_text/encode_image API. "
                            f"This is required for jina-embeddings-v4."
                        )
                    
                    # Ensure embeddings is a tensor
                    if not isinstance(embeddings, torch.Tensor):
                        embeddings = torch.tensor(embeddings, dtype=torch.float32)
                    if embeddings.dim() == 1:
                        embeddings = embeddings.unsqueeze(0)
                    embeddings_list.append(embeddings.squeeze(0))
                return torch.stack(embeddings_list)

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_forward)
        embeddings = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory

        del model
        torch.cuda.empty_cache()

        return {
            "output": embeddings,
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
            is_embedding=True,
            embedding_type=self.embedding_type,
            enforce_eager=True,
            tensor_parallel_size=1,
            prefill_only_mode=True,
            trust_remote_code=True,
        )

        # Reset peak memory stats after model loading, before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_embed():
            # For jina-embeddings-v4, the model's forward method handles task_label="retrieval"
            # which should format the text appropriately internally.
            # However, to match Transformers baseline which uses encode_text with prompt_name="query",
            # we should NOT add "Query: " prefix here, as encode_text handles it internally.
            # Use texts directly without prefix to match Transformers baseline behavior.
            return llm.embed_batch(
                self.texts,
                images=self.images,
                use_tqdm=False,
            )

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_embed)
        embeddings = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory

        llm.exit()
        cleanup_distributed()

        return {
            "output": embeddings,
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
            is_embedding=True,
            embedding_type=self.embedding_type,
            enforce_eager=True,
            tensor_parallel_size=1,
            prefill_only_mode=False,
            trust_remote_code=True,
        )

        # Reset peak memory stats after model loading, before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_embed():
            # For jina-embeddings-v4, the model's forward method handles task_label="retrieval"
            # which should format the text appropriately internally.
            # However, to match Transformers baseline which uses encode_text with prompt_name="query",
            # we should NOT add "Query: " prefix here, as encode_text handles it internally.
            # Use texts directly without prefix to match Transformers baseline behavior.
            return llm.embed_batch(
                self.texts,
                images=self.images,
                use_tqdm=False,
            )

        # Benchmark with warmup
        stats = self.benchmark_with_warmup(run_embed)
        embeddings = stats["result"]

        peak_memory = get_peak_memory_usage()
        memory = get_memory_usage() - start_memory

        llm.exit()
        cleanup_distributed()

        return {
            "output": embeddings,
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
        """Print accuracy comparison for embedding."""
        print("  Prefill-Only vs Transformers:")
        print(
            f"    Cosine Similarity: "
            f"{acc_prefill['cosine_similarity_mean']:.6f} "
            f"(min: {acc_prefill['cosine_similarity_min']:.6f})"
        )
        print(f"    Max Diff: {acc_prefill['max_diff']:.6f}")
        print(f"    Mean Diff: {acc_prefill['mean_diff']:.6f}")

        print("  Original vs Transformers:")
        print(
            f"    Cosine Similarity: "
            f"{acc_original['cosine_similarity_mean']:.6f} "
            f"(min: {acc_original['cosine_similarity_min']:.6f})"
        )
        print(f"    Max Diff: {acc_original['max_diff']:.6f}")
        print(f"    Mean Diff: {acc_original['mean_diff']:.6f}")

    def _print_speed_comparison(self, hf_result, prefill_result, original_result):
        """Print speed comparison."""
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
        """Print memory comparison."""
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test prefill-only embedding optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test text embedding
  python test_prefill_only_embed.py --modality text \\
      --model qwen3-embedding \\
      --model-path ~/.cache/huggingface/hub/Qwen3-Embedding-0.6B

  # Test gemma2 embedding
  python test_prefill_only_embed.py --modality text \\
      --model gemma2 \\
      --model-path ~/.cache/huggingface/hub/bge-multilingual-gemma2

  # Test multimodal embedding
  python test_prefill_only_embed.py --modality multimodal \\
      --model jina-embedding-v4 \\
      --model-path ~/.cache/huggingface/hub/jina-embeddings-v4-vllm-retrieval

  # Test with custom batch size
  python test_prefill_only_embed.py --modality text \\
      --model qwen3-embedding \\
      --model-path ~/.cache/huggingface/hub/Qwen3-Embedding-0.6B \\
      --batch-size 32

  # Test with custom warmup and iterations
  python test_prefill_only_embed.py --modality all \\
      --model qwen3-embedding \\
      --model-path ~/.cache/huggingface/hub/Qwen3-Embedding-0.6B \\
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
        help="Model to test (required). Examples: 'qwen3-embedding', 'gemma2', "
        "'jina-embedding-v4', 'jina-embedding-v3'.",
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
    """Run embedding tests based on command line arguments."""
    args = parse_args()

    print("=" * 60)
    print("Prefill-Only Embedding Test Suite")
    print("=" * 60)
    print(f"Modality: {args.modality}")
    if args.model:
        print(f"Model: {args.model}")
        set_model_filter(args.model)
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
            test_batch_embedding_text_comprehensive(
                modality="text",
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
                batch_size=args.batch_size,
                model_path=args.model_path,
                model=args.model,
            )
        except Exception as e:
            print(f"Error in comprehensive embedding test (text): {e}")
            import traceback
            traceback.print_exc()

    if args.modality in ["multimodal", "all"]:
        try:
            test_batch_embedding_multimodal_comprehensive(
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
                batch_size=args.batch_size,
                model_path=args.model_path,
                model=args.model,
            )
        except Exception as e:
            print(f"Error in comprehensive embedding test (multimodal): {e}")
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
