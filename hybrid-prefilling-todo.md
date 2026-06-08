# Hybrid Prefilling Implementation TODO

## Goal

Implement hybrid prefilling to reduce peak GPU memory from MLP intermediate activations, enabling much longer input sequences (e.g., long video understanding) on a single GPU.

## Core Idea

- MLP layers are token-independent: process them in chunks to bound peak activation memory
- Attention layers require full sequence context: process them normally
- Peak memory drops from `seq_len × intermediate_size` to `chunk_size × intermediate_size`

## Tasks

### 1. Add chunked MLP execution utility
- [ ] Create `nanovllm/layers/hybrid_prefill.py`
- [ ] Implement `chunked_mlp_forward(hidden_states, mlp_fn, chunk_size)` that splits input along sequence dim, applies MLP per chunk, and concatenates results
- [ ] Make chunk_size configurable (default 4096 tokens)

### 2. Integrate into model forward pass
- [ ] Modify `nanovllm/models/` decoder layer forward to use chunked MLP when `hybrid_prefill=True`
- [ ] Ensure attention still sees the full sequence (no chunking on attention)
- [ ] Gate behind config flag so existing behavior is unchanged by default

### 3. Add config support
- [ ] Add `hybrid_prefill: bool = False` to `Config`
- [ ] Add `hybrid_prefill_chunk_size: int = 4096` to `Config`
- [ ] Auto-enable when sequence length exceeds a threshold (optional)

### 4. Test correctness
- [ ] Verify output matches non-chunked forward (numerical equivalence)
- [ ] Test with varying sequence lengths (short, medium, long)
- [ ] Test with batch (varlen) inputs

### 5. Benchmark memory savings
- [ ] Measure peak memory with and without hybrid prefilling
- [ ] Test with increasing sequence lengths to show MIL expansion
