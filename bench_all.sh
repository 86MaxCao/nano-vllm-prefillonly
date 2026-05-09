#!/bin/bash
# ============================================================================
#  nano-vllm-prefillonly: Benchmark all supported models
#  Usage: bash bench_all.sh [GPU_ID] [CATEGORY]
#  Categories: embed, rerank, gen, vl_embed, vl_rerank, vl_gen, all
# ============================================================================

set -e

HF_CACHE=~/.cache/huggingface/hub/hf_cache
GPU=${1:-0}
CATEGORY=${2:-all}

export CUDA_VISIBLE_DEVICES=$GPU

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[BENCH]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }

# ============================================================================
#  Text Embedding
# ============================================================================
bench_embed() {
    log "=== Text Embedding ==="

    log "Qwen3-Embedding-0.6B"
    python3 -m examples.bench_prefillonly_embed \
        --modality text --batch-size 512 \
        --model qwen3-embedding \
        --model-path $HF_CACHE/Qwen3-Embedding-0.6B || warn "Qwen3-Embedding failed"

    log "Qwen3.5-Embedding-0.6B"
    python3 -m examples.bench_prefillonly_embed \
        --modality text --batch-size 512 \
        --model qwen3.5-embedding \
        --model-path $HF_CACHE/Qwen3.5-Embedding-0.6B || warn "Qwen3.5-Embedding failed"

    log "BGE-multilingual-gemma2"
    python3 -m examples.bench_prefillonly_embed \
        --modality text --batch-size 512 \
        --model gemma2 \
        --model-path $HF_CACHE/bge-multilingual-gemma2 || warn "BGE-Gemma2 failed"
}

# ============================================================================
#  Text Reranking
# ============================================================================
bench_rerank() {
    log "=== Text Reranking ==="

    log "Qwen3-Reranker-0.6B"
    python3 -m examples.bench_prefillonly_rerank \
        --modality text --batch-size 512 \
        --model qwen3-reranker \
        --model-path $HF_CACHE/Qwen3-Reranker-0.6B || warn "Qwen3-Reranker failed"

    log "Qwen3.5-Reranker-0.6B"
    python3 -m examples.bench_prefillonly_rerank \
        --modality text --batch-size 512 \
        --model qwen3.5-reranker \
        --model-path $HF_CACHE/Qwen3.5-Reranker-0.6B || warn "Qwen3.5-Reranker failed"

    log "Jina-Reranker-V3"
    python3 -m examples.bench_prefillonly_rerank \
        --modality text --batch-size 512 \
        --model jina-reranker-v3 \
        --model-path $HF_CACHE/jina-reranker-v3 || warn "Jina-Reranker-V3 failed"
}

# ============================================================================
#  Text Generation
# ============================================================================
bench_gen() {
    log "=== Text Generation ==="

    log "Qwen3-4B"
    python3 -m examples.bench_prefillonly_gen \
        --modality text \
        --model qwen3 \
        --model-path $HF_CACHE/Qwen3-4B || warn "Qwen3-4B failed"

    log "Qwen3.5-4B"
    python3 -m examples.bench_prefillonly_gen \
        --modality text \
        --model qwen3.5 \
        --model-path $HF_CACHE/Qwen3.5-4B || warn "Qwen3.5-4B failed"
}

# ============================================================================
#  VL Embedding
# ============================================================================
bench_vl_embed() {
    log "=== VL Embedding ==="

    log "Qwen3-VL-Embedding-2B"
    python3 -m examples.bench_prefillonly_embed \
        --modality multimodal \
        --model qwen3-vl-embedding \
        --model-path $HF_CACHE/Qwen3-VL-Embedding-2B || warn "Qwen3-VL-Embedding failed"

    log "Qwen3.5-VL-Embedding-2B"
    python3 -m examples.bench_prefillonly_embed \
        --modality multimodal \
        --model qwen3.5-vl-embedding \
        --model-path $HF_CACHE/Qwen3.5-VL-Embedding-2B || warn "Qwen3.5-VL-Embedding failed"

    log "Jina-Embeddings-V4"
    python3 -m examples.bench_prefillonly_embed \
        --modality multimodal \
        --model jina-embedding-v4 \
        --model-path $HF_CACHE/jina-embeddings-v4-vllm-retrieval || warn "Jina-V4 failed"
}

# ============================================================================
#  VL Reranking
# ============================================================================
bench_vl_rerank() {
    log "=== VL Reranking ==="

    log "Qwen3-VL-Reranker-2B"
    python3 -m examples.bench_prefillonly_rerank \
        --modality multimodal \
        --model qwen3-vl-reranker \
        --model-path $HF_CACHE/Qwen3-VL-Reranker-2B || warn "Qwen3-VL-Reranker failed"

    log "Qwen3.5-VL-Reranker-2B"
    python3 -m examples.bench_prefillonly_rerank \
        --modality multimodal \
        --model qwen3.5-vl-reranker \
        --model-path $HF_CACHE/Qwen3.5-VL-Reranker-2B || warn "Qwen3.5-VL-Reranker failed"

    log "Jina-Reranker-M0"
    python3 -m examples.bench_prefillonly_rerank \
        --modality multimodal \
        --model jina-reranker-m0 \
        --model-path $HF_CACHE/jina-reranker-m0 || warn "Jina-Reranker-M0 failed"
}

# ============================================================================
#  VL Generation
# ============================================================================
bench_vl_gen() {
    log "=== VL Generation ==="

    log "Qwen3-VL-2B"
    python3 -m examples.bench_prefillonly_gen \
        --modality multimodal --model qwen3vl \
        --model-path $HF_CACHE/Qwen3-VL-2B-Instruct || warn "Qwen3-VL failed"

    log "Qwen3.5-VL-2B"
    python3 -m examples.bench_prefillonly_gen \
        --modality multimodal --model qwen3.5vl \
        --model-path $HF_CACHE/Qwen3.5-VL-2B-Instruct || warn "Qwen3.5-VL failed"

    log "Qwen2-VL-2B"
    python3 -m examples.bench_prefillonly_gen \
        --modality multimodal --model qwen2vl \
        --model-path $HF_CACHE/Qwen2-VL-2B-Instruct || warn "Qwen2-VL failed"

    log "Qwen2.5-VL-3B"
    python3 -m examples.bench_prefillonly_gen \
        --modality multimodal --model qwen2.5vl \
        --model-path $HF_CACHE/Qwen2.5-VL-3B-Instruct || warn "Qwen2.5-VL failed"
}

# ============================================================================
#  Main
# ============================================================================
case "$CATEGORY" in
    embed)     bench_embed ;;
    rerank)    bench_rerank ;;
    gen)       bench_gen ;;
    vl_embed)  bench_vl_embed ;;
    vl_rerank) bench_vl_rerank ;;
    vl_gen)    bench_vl_gen ;;
    all)
        bench_embed
        bench_rerank
        bench_gen
        bench_vl_embed
        bench_vl_rerank
        bench_vl_gen
        ;;
    *)
        echo "Usage: bash bench_all.sh [GPU_ID] [CATEGORY]"
        echo "Categories: embed, rerank, gen, vl_embed, vl_rerank, vl_gen, all"
        exit 1
        ;;
esac

log "All benchmarks complete!"
