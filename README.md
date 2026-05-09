<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM Prefill-Only

A specialized optimization of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) for **prefill-only** inference tasks, designed for industrial-scale discriminative applications with multimodal large language models.

## 🎯 Why Prefill-Only?

Most real-world business scenarios are **prefill-only tasks**, especially in discriminative applications:

- **Reranking**: Determining the relevance of documents to queries
- **Retrieval/Embedding**: Generating vector representations for semantic search
- **Classification**: Binary or multi-class classification tasks
- **Visual Question Answering**: Answering yes/no questions about images
- **Spatial Reasoning**: Comparing object sizes, positions, or relationships
- **Attribute Recognition**: Identifying colors, shapes, or other visual attributes

### Industrial Applications

In industrial settings, multimodal LLMs are increasingly replacing traditional discriminative vision models for tasks like:

- **Object Detection Queries**: "Is there a dog in this image?" (Binary classification)
- **Spatial Comparisons**: "Which object is larger?" or "Which is on the left?"
- **Attribute Classification**: "What color is the building?" (Red/Black/White/Blue/Yellow/Green/Gray)
- **Multimodal Retrieval**: Finding the most relevant image from a large collection, e.g., "Find the image that best represents traditional Chinese architecture" from thousands of building photos
- **Multimodal Reranking**: Ranking images by relevance, e.g., "Which shop sign is most eye-catching?" from a collection of street photos, or "Which product image best matches the query description?"

When processing **hundreds of millions of images** at scale, the traditional vLLM approach with KV cache becomes inefficient. Our prefill-only optimization eliminates unnecessary KV cache allocation and management overhead, significantly improving inference efficiency for single-token generation tasks.

## 🚀 Key Features

* ⚡ **Optimized for Single-Token Generation** - No KV cache overhead for discriminative tasks
* 💾 **Massive Memory Savings** - Up to **10x less memory** compared to original nano-vllm
* 🎯 **Industrial-Scale Ready** - Designed for high-throughput discriminative inference
* 🔧 **Based on nano-vllm** - Built on top of the clean, readable nano-vllm codebase

## 📊 Performance Benchmarks

### 1. Qwen3-VL-2B-Instruct (Multimodal Generation)

**Test Configuration:**
- **Model**: Qwen3-VL-2B-Instruct
- **Task**: Multimodal single-token generation
- **Hardware**: Single H20 GPU

**Speed Comparison:**

| Inference Engine | Mean (s) | Median (s) | P90 (s) | P99 (s) | Speedup vs Transformers |
|-----------------|----------|------------|---------|---------|------------------------|
| Transformers     | 1.2112   | 1.2121     | 1.2716  | 1.2738  | 1.00x (baseline)       |
| Prefill-Only     | 0.5766   | 0.5349     | 0.7640  | 0.8717  | **2.10x faster** ⚡     |
| Original nano-vllm | 0.5712 | 0.5667     | 0.6211  | 0.6246  | 2.12x faster            |

**Memory Comparison (Peak):**

| Metric               | Transformers | Prefill-Only | Original nano-vllm | Prefill vs Original |
|---------------------|--------------|--------------|---------------------|---------------------|
| Peak Allocated      | 4459.80 MB   | 4892.34 MB   | 49680.06 MB         | **10.15x less** 💾  |
| Peak Reserved       | 4744.00 MB   | 5226.00 MB   | 49998.00 MB         | 9.57x less          |
| Current Allocated   | 4090.96 MB   | 4218.46 MB   | 42467.46 MB         | 10.07x less         |
| Current Reserved    | 4744.00 MB   | 4308.00 MB   | 49998.00 MB         | 11.61x less         |
| Fragmentation %     | 0.15%        | 2.08%        | 0.55%               | -                   |

**Key Insight**: While prefill-only mode is slightly slower (0.99x) than original nano-vllm, it uses **only 10% of the memory**, making it ideal for high-throughput scenarios where memory efficiency is critical.

---

### 2. Qwen3-4B (Text Generation)

**Test Configuration:**
- **Model**: Qwen3-4B
- **Task**: Text-only single-token generation

**Speed Comparison:**

| Inference Engine | Mean (s) | Median (s) | P90 (s) | P99 (s) | Speedup vs Transformers |
|-----------------|----------|------------|---------|---------|------------------------|
| Transformers     | 0.2367   | 0.2366     | 0.2377  | 0.2416  | 1.00x (baseline)       |
| Prefill-Only     | 0.1352   | 0.1271     | 0.1568  | 0.2653  | **1.75x faster** ⚡     |
| Original         | 0.1264   | 0.1248     | 0.1254  | 0.1311  | 1.87x faster            |

**Memory Comparison (Peak):**

| Metric               | Transformers | Prefill-Only | Original | Prefill vs Original |
|---------------------|--------------|--------------|----------|---------------------|
| Peak Memory         | 9040.87 MB   | 8007.38 MB   | 85825.38 MB | **10.72x less** 💾  |

---

### 3. Qwen3-Embedding-0.6B (Embedding Model)

**Test Configuration:**
- **Model**: Qwen3-Embedding-0.6B
- **Task**: Text embedding for semantic search

**Accuracy Comparison:**
- Prefill-Only vs Transformers: Cosine Similarity: **0.999761** (min: 0.999707), Max Diff: 0.003435, Mean Diff: 0.000533
- Original vs Transformers: Cosine Similarity: **0.999761** (min: 0.999707), Max Diff: 0.003435, Mean Diff: 0.000533

**Speed Comparison:**

| Inference Engine | Mean (s) | Median (s) | P90 (s) | P99 (s) | Speedup vs Transformers |
|-----------------|----------|------------|---------|---------|------------------------|
| Transformers     | 0.0366   | 0.0363     | 0.0373  | 0.0374  | 1.00x (baseline)       |
| Prefill-Only     | 0.0246   | 0.0245     | 0.0257  | 0.0261  | **1.49x faster** ⚡     |
| Original         | 0.0247   | 0.0248     | 0.0252  | 0.0257  | 1.48x faster            |

**Memory Comparison (Peak):**

| Metric               | Transformers | Prefill-Only | Original | Prefill vs Original |
|---------------------|--------------|--------------|----------|---------------------|
| Peak Memory         | 1190.87 MB   | 1192.38 MB   | 2316.02 MB | 1.94x less 💾       |

---

### 4. bge-multilingual-gemma2 (Multilingual Embedding Model)

**Test Configuration:**
- **Model**: bge-multilingual-gemma2
- **Task**: Multilingual text embedding

**Accuracy Comparison:**
- Prefill-Only vs Transformers: Cosine Similarity: **0.999973** (min: 0.999939), Max Diff: 0.000886, Mean Diff: 0.000091
- Original vs Transformers: Cosine Similarity: **0.999973** (min: 0.999939), Max Diff: 0.000886, Mean Diff: 0.000091

**Speed Comparison:**

| Inference Engine | Mean (s) | Median (s) | P90 (s) | P99 (s) | Speedup vs Transformers |
|-----------------|----------|------------|---------|---------|------------------------|
| Transformers     | 0.0715   | 0.0713     | 0.0717  | 0.0730  | 1.00x (baseline)       |
| Prefill-Only     | 0.0449   | 0.0446     | 0.0463  | 0.0466  | **1.59x faster** ⚡     |
| Original         | 0.0452   | 0.0430     | 0.0455  | 0.0631  | 1.58x faster            |

**Memory Comparison (Peak):**

| Metric               | Transformers | Prefill-Only | Original | Prefill vs Original |
|---------------------|--------------|--------------|----------|---------------------|
| Peak Memory         | 17677.26 MB  | 17677.25 MB  | 35303.30 MB | **2.00x less** 💾  |

> **Note**: Further performance optimizations are under active development. The current implementation focuses on correctness and memory efficiency, with speed optimizations planned for future releases.

## 📦 Installation

### Prerequisites

Install the required dependencies (same as nano-vllm):

```bash
# Clone the repository
git clone https://github.com/86MaxCao/nano-vllm-prefillonly.git
cd nano-vllm-prefillonly

# Install dependencies (same as nano-vllm)
pip install -r requirements.txt
```

> **Note**: If you already have nano-vllm installed, you may not need to install additional packages. The dependencies are identical.

### Running Without Package Installation

You can run the benchmarks directly without installing the package itself (no `pip install .` needed):

```bash
# Run directly using Python module syntax
python3 -m examples.bench_prefillonly_gen --modality multimodal --model qwen3vl --model-path ~/.cache/huggingface/hub/Qwen3-VL-2B-Instruct
```

> **Note**: For developers who need to install the package itself, you can modify `pyproject.toml` and use `pip install .` as needed.

## 🎮 Quick Start

All models use three benchmark scripts depending on task type:

| Task Type | Script | Key Arguments |
|-----------|--------|---------------|
| Generation | `examples.bench_prefillonly_gen` | `--modality`, `--model`, `--model-path` |
| Embedding | `examples.bench_prefillonly_embed` | `--modality`, `--model`, `--model-path` |
| Reranking | `examples.bench_prefillonly_rerank` | `--modality`, `--model`, `--model-path` |

> **Note**: `--model-path` points to the local model directory (e.g., HuggingFace cache). Set `HF_CACHE=~/.cache/huggingface/hub/hf_cache` or adjust paths to match your setup.

### Text Generation

#### Qwen3

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_gen \
    --modality text \
    --model qwen3 \
    --model-path $HF_CACHE/Qwen3-4B
```

#### Qwen3.5

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_gen \
    --modality text \
    --model qwen3.5 \
    --model-path $HF_CACHE/Qwen3.5-4B
```

### Multimodal Generation

#### Qwen3-VL

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_gen \
    --modality multimodal \
    --model qwen3vl \
    --model-path $HF_CACHE/Qwen3-VL-2B-Instruct
```

#### Qwen3.5

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_gen \
    --modality multimodal \
    --model qwen3.5vl \
    --model-path $HF_CACHE/Qwen3.5-4B
```

#### Qwen2-VL

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_gen \
    --modality multimodal \
    --model qwen2vl \
    --model-path $HF_CACHE/Qwen2-VL-2B-Instruct
```

#### Qwen2.5-VL

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_gen \
    --modality multimodal \
    --model qwen2.5vl \
    --model-path $HF_CACHE/Qwen2.5-VL-3B-Instruct
```

### Text Embedding

#### Qwen3-Embedding

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_embed \
    --modality text \
    --model qwen3-embedding \
    --model-path $HF_CACHE/Qwen3-Embedding-0.6B
```

#### BGE-multilingual-gemma2

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_embed \
    --modality text \
    --model gemma2 \
    --model-path $HF_CACHE/bge-multilingual-gemma2
```

### Multimodal Embedding

#### Qwen3-VL-Embedding

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_embed \
    --modality multimodal \
    --model qwen3-vl-embedding \
    --model-path $HF_CACHE/Qwen3-VL-Embedding-2B
```

### Text Reranking

#### Qwen3-Reranker

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_rerank \
    --modality text \
    --model qwen3-reranker \
    --model-path $HF_CACHE/Qwen3-Reranker-0.6B
```

#### Jina-Reranker-V3

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_rerank \
    --modality text \
    --model jina-reranker-v3 \
    --model-path $HF_CACHE/jina-reranker-v3
```

#### BGE-Reranker-Gemma

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_rerank \
    --modality text \
    --model bge-reranker-v2-gemma \
    --model-path $HF_CACHE/bge-reranker-v2-gemma
```

### Multimodal Reranking

#### Qwen3-VL-Reranker

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_rerank \
    --modality multimodal \
    --model qwen3-vl-reranker \
    --model-path $HF_CACHE/Qwen3-VL-Reranker-2B
```

#### Jina-Reranker-M0

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_rerank \
    --modality multimodal \
    --model jina-reranker-m0 \
    --model-path $HF_CACHE/jina-reranker-m0
```

### Batch Benchmarking All Models

Use `bench_all.sh` to benchmark all supported models at once:

```bash
# Benchmark all models on GPU 0
bash bench_all.sh 0 all

# Benchmark specific categories
bash bench_all.sh 0 embed       # Text embedding only
bash bench_all.sh 0 rerank      # Text reranking only
bash bench_all.sh 0 gen         # Text generation only
bash bench_all.sh 0 vl_embed    # VL embedding only
bash bench_all.sh 0 vl_rerank   # VL reranking only
bash bench_all.sh 0 vl_gen      # VL generation only
```

## 📝 Current Status

**What Works:**
- ✅ Single-token generation for text-only models (Qwen3, Qwen3.5)
- ✅ Single-token generation for multimodal models (Qwen3-VL, Qwen3.5, Qwen2-VL, Qwen2.5-VL)
- ✅ Text embedding (Qwen3-Embedding, BGE-Gemma2)
- ✅ Multimodal embedding (Qwen3-VL-Embedding)
- ✅ Text reranking (Qwen3-Reranker, Jina-Reranker-V3, BGE-Reranker-Gemma)
- ✅ Multimodal reranking (Qwen3-VL-Reranker, Jina-Reranker-M0)
- ✅ Memory-efficient inference without KV cache
- ✅ Accuracy matching with Transformers baseline

**Not Yet Supported:**
- ❌ LLaVA-NeXT prefill-only (tensor shape mismatch in multimodal projector)
- ❌ Jina-Embedding-V4 (custom modeling code incompatible with current transformers)
- ❌ GME-Qwen2-VL (requires older transformers version)

### Supported Models & VRAM Savings

Our prefill-only optimization skips KV cache allocation entirely for embedding and reranker models, while vLLM allocates KV cache for ALL models (even when completely unused). On a 96GB H20 GPU, the KV cache alone can consume 82-85GB — meaning vLLM can only serve one embedding/reranker model per GPU, while our framework uses just the model weights.

| Model | Category | Prefill-Only VRAM | vLLM-Style VRAM | KV Cache | VRAM Saved |
|-------|----------|-------------------|-----------------|----------|------------|
| Qwen3-0.6B | text_gen | 1,703 MB | 86,507 MB | 85,288 MB | **98.0%** |
| Qwen3-VL-2B | vl_gen | 4,812 MB | 52,196 MB | - | **90.8%** |
| Qwen2.5-VL-3B | vl_gen | 11,005 MB | 62,234 MB | - | **82.3%** |
| Qwen3-Embedding-0.6B | text_embed | 1,177 MB | 86,472 MB | 85,288 MB | **98.6%** |
| BGE-Gemma2 | text_embed | 17,665 MB | 87,469 MB | 69,804 MB | **79.8%** |
| Qwen3-Reranker-0.6B | text_rerank | 1,453 MB | 86,168 MB | 84,980 MB | **98.3%** |
| Jina-Reranker-V3 | text_rerank | 1,234 MB | 86,157 MB | 84,924 MB | **98.6%** |
| BGE-Reranker-Gemma | text_rerank | 4,818 MB | 87,542 MB | 82,724 MB | **94.5%** |
| Qwen3-VL-Embedding-2B | vl_embed | 4,780 MB | 85,706 MB | 82,264 MB | **94.4%** |
| Qwen3-VL-Reranker-2B | vl_rerank | 4,844 MB | 85,706 MB | 82,264 MB | **94.3%** |
| Jina-Reranker-M0 | vl_rerank | 4,661 MB | 87,541 MB | 82,880 MB | **94.7%** |

#### How VRAM is Measured

- **Prefill-Only VRAM**: Peak GPU memory allocated when loading and running inference with our framework (no KV cache). Measured via `torch.cuda.max_memory_allocated()` in an isolated subprocess to avoid CUDA memory pool contamination between models.
- **vLLM-Style VRAM**: Simulates vLLM's behavior, which allocates KV cache even for embedding/reranker models. For models that can be loaded as generation models, we load them as such to force KV cache allocation and measure the peak. For models that cannot (e.g., VL rerankers), we compute the KV cache size using the same formula as vLLM: `available_memory = total_gpu_memory × gpu_memory_utilization - model_weights`, then add it to the model weight VRAM.
- **KV Cache**: The size of the KV cache tensor alone, computed as `2 × num_layers × block_size × num_kv_heads × head_dim × dtype_bytes × num_blocks`. This memory is completely wasted for embedding/reranker models since they only need a single forward pass.
- All measurements taken on NVIDIA H20 (96GB) GPUs with subprocess isolation (each model measured in a fresh process to avoid cross-model memory contamination).

**In Progress:**
- 🔄 LLaVA-NeXT prefill-only (tensor shape mismatch in multimodal projector)
- 🔄 Jina-Embedding-V4 (custom modeling code incompatible with current transformers)
- 🔄 GME-Qwen2-VL (requires older transformers version)
- 🔄 Production-ready features and detailed demos

## 🏗️ Architecture

This project is built on top of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), a lightweight vLLM implementation. The prefill-only optimization:

1. **Skips KV Cache Allocation**: For single-token generation, KV cache is unnecessary
2. **Uses Fallback Path for Vision**: Directly processes `pixel_values` without vision cache
3. **Optimizes Memory Usage**: Eliminates cache management overhead

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project inherits the license from [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## 🙏 Acknowledgments

- Built on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) by [GeeeekExplorer](https://github.com/GeeeekExplorer)
- Inspired by the need for efficient discriminative inference at scale
