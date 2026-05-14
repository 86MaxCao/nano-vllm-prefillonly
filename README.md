<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM Prefill-Only

A specialized optimization of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) for **prefill-only** inference tasks, designed for industrial-scale discriminative applications with multimodal large language models.

> **Motivation**: This project addresses the problem described in [vllm-project/vllm#29584](https://github.com/vllm-project/vllm/issues/29584) — vLLM unconditionally allocates KV cache even for non-autoregressive tasks (embedding, reranking, classification), wasting up to **85-98% of GPU memory** on completely unused cache tensors. The vLLM maintainers acknowledged this issue but noted that fixing it "would require modifications to a lot of core code" and closed it as not planned. Our framework solves this by **completely eliminating KV cache allocation** for prefill-only workloads, enabling single-GPU deployment of models that would otherwise require multi-GPU setups under vLLM.

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

All benchmarks measured on a single NVIDIA H20 GPU (96GB). Speed measured as mean latency over 5 iterations after warmup. VRAM measured via `torch.cuda.max_memory_allocated()` in isolated subprocesses.

### Speed Comparison: Text Models (batch=100)

| Model | Category | Transformers (s) | Prefill-Only (s) | Speedup |
|-------|----------|:-----------------:|:-----------------:|:-------:|
| Qwen3-0.6B | Generation | 0.0524 | 0.0309 | **1.70x** |
| Qwen3-Embedding-0.6B | Embedding | 0.0330 | 0.0242 | **1.36x** |
| bge-multilingual-gemma2 | Embedding | 0.2613 | 0.1595 | **1.64x** |
| Qwen3-Reranker-0.6B | Reranking | 0.1574 | 0.0610 | **2.58x** |
| bge-reranker-v2-gemma | Reranking | 0.1736 | 0.1162 | **1.49x** |


### Speed Comparison: Multimodal Models (batch=10, 224x224 images)

| Model | Category | Transformers (s) | Prefill-Only (s) | Speedup |
|-------|----------|:-----------------:|:-----------------:|:-------:|
| Qwen3-VL-2B-Instruct | Generation | 0.0864 | 0.0499 | **1.73x** |
| Qwen2.5-VL-3B-Instruct | Generation | 0.1212 | 0.0594 | **2.04x** |
| Qwen3-VL-Embedding-2B | Embedding | 0.0704 | 0.0651 | **1.08x** |
| Qwen3-VL-Reranker-2B | Reranking | 0.0829 | 0.0738 | **1.12x** |

> Both Transformers and Prefill-Only use FlashAttention. End-to-end measurement includes preprocessing (tokenization/apply_chat_template + image processing).

### Why Varlen Attention is Faster (Forward-Only Comparison)

Our framework uses **varlen FlashAttention** (`flash_attn_varlen_func`), which concatenates all sequences into a single 1D tensor and uses `cu_seqlens` to delineate boundaries. This eliminates padding waste that Transformers' standard batch attention suffers from.

**Benchmark**: Qwen3-VL-2B-Instruct, model forward only (excluding preprocessing), NVIDIA H20.

| Scenario | Transformers | Prefill-Only | Speedup | Why |
|----------|:------------:|:------------:|:-------:|-----|
| 10 images, same size (224px) | 67 ms | 66 ms | 1.02x | No padding difference, nearly identical |
| 100 images, same size (224px) | 565 ms | 534 ms | 1.06x | Slight advantage from avoiding attention_mask overhead |
| 10 images, variable size (224-672px) | 264 ms | 175 ms | **1.50x** | HF pads to max (454 tokens), 48% tokens wasted |

**Key insight**: The speedup scales with **sequence length variance**. When images have different resolutions, Transformers pads all sequences to the longest one, wasting compute on padding tokens. Our varlen approach computes only real tokens — the more diverse the input lengths, the bigger the advantage.

In production scenarios with mixed-resolution images (common in real-world applications), this translates to 1.3-1.5x faster model forward passes with zero accuracy loss.

### torch.compile Status

Currently, our framework applies `@torch.compile` to **individual operators** (RMSNorm, SiLU, RoPE, Sampler) but does NOT perform **full-graph compilation** or **cross-operator kernel fusion**. In contrast, vLLM v1 compiles the entire model forward pass as a single graph, enabling TorchInductor to fuse adjacent operators (e.g., RMSNorm output directly into linear input) and reduce memory bandwidth by ~20%.

This is a known optimization gap. Full-graph `torch.compile` support for prefill-only workloads is planned for a future release.

### VRAM Savings (Prefill-Only vs vLLM-Style KV Cache)

Our framework **completely eliminates KV cache allocation** for prefill-only workloads, while vLLM allocates KV cache for ALL models (even when entirely unused for embedding/reranking). On a 96GB H20 GPU, the KV cache alone can consume 40-85GB.

| Model | Category | Prefill-Only VRAM | vLLM-Style VRAM | VRAM Saved |
|-------|----------|:-----------------:|:---------------:|:----------:|
| Qwen3-0.6B | Text Generation | 1,703 MB | 86,507 MB | **98.0%** |
| Qwen3-Embedding-0.6B | Text Embedding | 1,177 MB | 86,472 MB | **98.6%** |
| bge-multilingual-gemma2 | Text Embedding | 17,665 MB | 87,469 MB | **79.8%** |
| Qwen3-Reranker-0.6B | Text Reranking | 1,453 MB | 86,168 MB | **98.3%** |
| bge-reranker-v2-gemma | Text Reranking | 4,818 MB | 87,542 MB | **94.5%** |
| Qwen3-VL-2B-Instruct | Multimodal Generation | 4,812 MB | 52,196 MB | **90.8%** |
| Qwen2.5-VL-3B-Instruct | Multimodal Generation | 11,005 MB | 62,234 MB | **82.3%** |
| Qwen3-VL-Embedding-2B | Multimodal Embedding | 4,780 MB | 85,706 MB | **94.4%** |
| Qwen3-VL-Reranker-2B | Multimodal Reranking | 4,844 MB | 85,706 MB | **94.3%** |

#### How VRAM is Measured

- **Prefill-Only VRAM**: Peak GPU memory allocated when loading and running inference with our framework (no KV cache). Measured via `torch.cuda.max_memory_allocated()` in isolated subprocesses.
- **vLLM-Style VRAM**: Simulates vLLM's behavior of allocating KV cache even for embedding/reranker models. For models loadable as generation models, we force KV cache allocation and measure. For others, KV cache size is computed using vLLM's formula and added to model weight VRAM.
- All measurements taken on NVIDIA H20 (96GB) with subprocess isolation.

## 📦 Installation

```bash
git clone https://github.com/86MaxCao/nano-vllm-prefillonly.git
cd nano-vllm-prefillonly
pip install -e .
```

**Dependencies**: Python 3.10+, PyTorch 2.4+, Transformers 4.51+, Flash-Attention, Triton.

## 🎮 Quick Start

### Text Generation

```python
from nanovllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-0.6B")
sp = SamplingParams(max_tokens=1)

prompts = ["Is the Earth round? Answer Yes or No."] * 100
for p in prompts:
    llm.add_request(p, sp)

outputs = {}
while not llm.is_finished():
    output, _ = llm.step()
    for seq_id, token_ids in output:
        outputs[seq_id] = token_ids
```

### Multimodal Generation

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image

llm = LLM("Qwen/Qwen3-VL-2B-Instruct", multimodal_model_type="qwen3_vl")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
sp = SamplingParams(max_tokens=1)

images = [Image.open(f"image_{i}.jpg") for i in range(10)]
requests = [
    {
        "messages": [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "What is in this image? Answer in one word."},
        ]}],
        "images": [img],
    }
    for img in images
]

results = llm.generate_multimodal(requests, sp, processor)
```

### Text Embedding

```python
from nanovllm import LLM

llm = LLM("Qwen/Qwen3-Embedding-0.6B", is_embedding=True)

texts = ["What is deep learning?", "Explain transformers", "What is NLP?"]
embeddings = llm.embed_batch(texts)  # [batch_size, hidden_size]
```

### Multimodal Embedding

```python
from nanovllm import LLM
from PIL import Image

llm = LLM("Qwen/Qwen3-VL-Embedding-2B", multimodal_model_type="qwen3_vl", is_embedding=True)

images = [Image.open("photo.jpg")]
texts = ["A photo of a building"]
embeddings = llm.embed_batch(texts, images=images)  # [batch_size, hidden_size]
```

### Text Reranking

```python
from nanovllm import LLM

llm = LLM("Qwen/Qwen3-Reranker-0.6B", is_reranker=True)

pairs = [
    ("What is AI?", "Artificial intelligence is the simulation of human intelligence."),
    ("What is Python?", "Python is a programming language."),
]
scores = llm.rerank_batch(pairs)  # [batch_size]
```

### Multimodal Reranking

```python
from nanovllm import LLM
from PIL import Image

llm = LLM("Qwen/Qwen3-VL-Reranker-2B", multimodal_model_type="qwen3_vl", is_reranker=True)

images = [Image.open("photo.jpg")]
pairs = [("Find a building", "A document about architecture")]
scores = llm.rerank_batch(pairs, images=images)  # [batch_size]
```

## 🏗️ Supported Models

| Model | Type | `multimodal_model_type` | Extra Args |
|-------|------|------------------------|------------|
| Qwen3-0.6B | Text Generation | - | - |
| Qwen3-Embedding-0.6B | Text Embedding | - | `is_embedding=True` |
| bge-multilingual-gemma2 | Text Embedding | - | `is_embedding=True` |
| Qwen3-Reranker-0.6B | Text Reranking | - | `is_reranker=True` |
| bge-reranker-v2-gemma | Text Reranking | - | `is_reranker=True` |
| Qwen3-VL-2B-Instruct | Multimodal Generation | `qwen3_vl` | - |
| Qwen2.5-VL-3B-Instruct | Multimodal Generation | `qwen2_5_vl` | - |
| Qwen3-VL-Embedding-2B | Multimodal Embedding | `qwen3_vl` | `is_embedding=True` |
| Qwen3-VL-Reranker-2B | Multimodal Reranking | `qwen3_vl` | `is_reranker=True` |

> `multimodal_model_type` can be auto-detected from model path (e.g., paths containing "qwen3" + "vl" auto-resolve to `qwen3_vl`). Explicit specification is only needed when auto-detection fails.

## 📝 Current Status

**What Works:**
- Single-token generation for text and multimodal models
- Text and multimodal embedding with batch processing
- Text and multimodal reranking with batch processing
- Memory-efficient inference without KV cache
- Accuracy matching with Transformers baseline

**Known Limitations:**
- jina-reranker-v3: Custom bidirectional architecture, causal varlen path adds overhead
- Full-graph `torch.compile` not yet supported (only per-operator compile)

## 🏗️ Architecture

Built on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). Key optimizations:

1. **No KV Cache** - Eliminates cache allocation entirely for prefill-only tasks
2. **Varlen FlashAttention** - Concatenates sequences into 1D tensor, eliminates padding waste
3. **Batch Preprocessing** - Batch `apply_chat_template` + batch processor calls

## 📄 License

MIT License (inherited from [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)).

## 🙏 Acknowledgments

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) by [GeeeekExplorer](https://github.com/GeeeekExplorer)
