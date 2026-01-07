#!/bin/bash

# Test script for nano-vllm-prefillonly
# This script runs various tests for embedding, reranking, and generation tasks





# ============================================================================
#                    Text-only tests
# ============================================================================

# # -------- Test embedding task (text modality) --------------------
CUDA_VISIBLE_DEVICES=1 python3 -m examples.bench_prefillonly_embed \
    --modality text \
    --batch-size 512 \
    --model qwen3-embedding \
    --model-path ~/.cache/huggingface/hub/Qwen3-Embedding-0.6B


CUDA_VISIBLE_DEVICES=1 python3 -m examples.bench_prefillonly_embed \
    --modality text \
    --batch-size 512 \
    --model gemma2 \
    --model-path ~/.cache/huggingface/hub/bge-multilingual-gemma2


# # -------- Test reranking task (text modality) --------------------
CUDA_VISIBLE_DEVICES=2 python3 -m examples.bench_prefillonly_rerank \
    --modality text \
    --batch-size 512 \
    --model qwen3-reranker \
    --model-path ~/.cache/huggingface/hub/Qwen3-Reranker-0.6B


CUDA_VISIBLE_DEVICES=2 python3 -m examples.bench_prefillonly_rerank \
    --modality text \
    --batch-size 512 \
    --model jina-reranker-v3 \
    --model-path ~/.cache/huggingface/hub/jina-reranker-v3


CUDA_VISIBLE_DEVICES=2 python3 -m examples.bench_prefillonly_rerank \
    --modality text \
    --batch-size 512 \
    --model bge-reranker-v2-gemma \
    --model-path ~/.cache/huggingface/hub/bge-reranker-v2-gemma


# # -------- Test generation task (text modality) --------------------
CUDA_VISIBLE_DEVICES=3 python3 -m examples.bench_prefillonly_gen \
    --modality text \
    --model qwen3 \
    --model-path ~/.cache/huggingface/hub/Qwen3-0.6B



# ============================================================================
#                         Multimodal tests
# ============================================================================

# # -------- Test specific embedding models -------------------- 
CUDA_VISIBLE_DEVICES=1 python3 -m examples.bench_prefillonly_embed \
    --modality multimodal \
    --model jina-embedding-v4 \
    --model-path ~/.cache/huggingface/hub/jina-embeddings-v4-vllm-retrieval


# # -------- Test specific reranker model --------------------
CUDA_VISIBLE_DEVICES=2 python3 -m examples.bench_prefillonly_rerank \
    --modality multimodal \
    --model jina-reranker-m0 \
    --model-path ~/.cache/huggingface/hub/jina-reranker-m0


# # -------- Test specific generation model -------------------- 
CUDA_VISIBLE_DEVICES=3 python3 -m examples.bench_prefillonly_gen \
    --modality multimodal \
    --model llavanext \
    --model-path ~/.cache/huggingface/hub/llava-v1.6-mistral-7b-hf
    

CUDA_VISIBLE_DEVICES=3 python3 -m examples.bench_prefillonly_gen \
    --modality multimodal \
    --model qwen2vl \
    --model-path ~/.cache/huggingface/hub/Qwen2-VL-2B-Instruct


CUDA_VISIBLE_DEVICES=3 python3 -m examples.bench_prefillonly_gen \
    --modality multimodal \
    --model qwen2.5vl \
    --model-path ~/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct


CUDA_VISIBLE_DEVICES=3 python3 -m examples.bench_prefillonly_gen \
    --modality multimodal \
    --model qwen3vl \
    --model-path ~/.cache/huggingface/hub/Qwen3-VL-2B-Instruct

