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

### Multimodal Single-Token Generation

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_gen \
    --modality multimodal \
    --model qwen3vl \
    --model-path ~/.cache/huggingface/hub/Qwen3-VL-2B-Instruct
```

### Text-Only Single-Token Generation

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_gen \
    --modality text \
    --model qwen3 \
    --model-path ~/.cache/huggingface/hub/Qwen3-4B
```

### Text Embedding

#### Qwen3-Embedding

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_embed \
    --modality text \
    --model qwen3-embedding \
    --model-path ~/.cache/huggingface/hub/Qwen3-Embedding-0.6B
```

#### bge-multilingual-gemma2

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m examples.bench_prefillonly_embed \
    --modality text \
    --model gemma2 \
    --model-path ~/.cache/huggingface/hub/bge-multilingual-gemma2
```

> **Note**: Detailed usage examples and demos will be provided in future releases. For now, refer to the benchmark scripts in `examples/` directory.

## 📝 Current Status

⚠️ **Early Development**: This project is currently in active development. The core functionality for prefill-only inference has been implemented and tested, but the codebase is still being optimized and refined.

**What Works:**
- ✅ Single-token generation for text-only models
- ✅ Single-token generation for multimodal models (Qwen3-VL, Qwen2-VL, Qwen2.5-VL, LLaVA-NeXT)
- ✅ Memory-efficient inference without KV cache
- ✅ Accuracy matching with Transformers baseline

**Currently Supported Models:**
- ✅ **Qwen3**: Text generation models (e.g., Qwen3-4B, Qwen3-0.6B)
- ✅ **Qwen3-VL**: Multimodal generation models (e.g., Qwen3-VL-2B-Instruct)
- ✅ **Qwen3-Embedding**: Embedding models for semantic search (e.g., Qwen3-Embedding-0.6B)
- ✅ **bge-multilingual-gemma2**: Multilingual embedding models (e.g., bge-multilingual-gemma2)

**In Progress:**
- 🔄 **Model Support**: Adapting all 12 models from `bench_all.sh` (currently in development)
  - Text-only: Qwen3-Reranker, Jina-Reranker-V3, BGE-Reranker-V2-Gemma
  - Multimodal: Jina-Embedding-V4, Jina-Reranker-M0, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL
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
