# LLM Inference Learning Path

## Goal

Build an LLM inference engine from scratch in C++, load existing trained weights, and verify its numerical output against a trusted reference implementation. After the CPU implementation is correct, add a Vulkan compute backend and study modern inference techniques.

The first target is GPT-2 124M. It is small enough to run locally, has publicly available weights, and uses the same fundamental decoder-only Transformer structure as GPT-3.

## Guiding Principles

- Prefer one complete implementation over many incomplete experiments.
- Establish correctness on the CPU before optimizing or using the GPU.
- Compare logits and intermediate tensors, not only generated text.
- Keep the first implementation explicit. Avoid building a general-purpose tensor framework.
- Add one feature at a time and preserve a known-correct execution path.
- Treat tokenization, model inference, sampling, and optimization as separate subsystems.
- Postpone training until inference and numerical validation are well understood.

## Target Architecture

The initial model is GPT-2 124M:

| Property | Value |
| --- | ---: |
| Transformer layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Context length | 1,024 |
| Vocabulary size | 50,257 |
| Approximate parameter count | 124 million |

The basic inference path is:

```text
text
  -> byte-level BPE tokenizer
  -> token and position embeddings
  -> repeated Transformer blocks
       -> LayerNorm
       -> causal multi-head self-attention
       -> residual connection
       -> LayerNorm
       -> MLP with GELU
       -> residual connection
  -> final LayerNorm
  -> vocabulary projection
  -> logits
  -> token selection
  -> next token
```

## Phase 1: Establish the Reference

Use Karpathy's [`llm.c`](https://github.com/karpathy/llm.c) as the executable reference, especially its readable FP32 CPU implementation in [`train_gpt2.c`](https://github.com/karpathy/llm.c/blob/master/train_gpt2.c).

Tasks:

1. Build and run the CPU reference.
2. Obtain the prepared GPT-2 124M checkpoint and tokenizer artifacts.
3. Run a fixed sequence of token IDs through the model.
4. Record the final logits and selected intermediate activations as golden test data.
5. Fix the model configuration, prompt tokens, precision, and compiler settings used for comparisons.

The reference exists to answer numerical questions. The new C++ implementation should be written independently rather than produced by translating the C source line by line.

Completion criteria:

- The reference runs locally without a Python runtime.
- A fixed input produces repeatable logits.
- The checkpoint layout and tensor dimensions are documented.

## Phase 2: Implement the FP32 CPU Backend

Use contiguous FP32 buffers and small tensor views that describe dimensions and strides. Do not create autograd, broadcasting, a graph executor, or a general tensor library.

Implement and unit-test these primitives first:

- Dot product
- Matrix-vector and matrix-matrix multiplication
- Elementwise addition and multiplication
- Mean and variance reduction
- LayerNorm
- GELU
- Numerically stable softmax
- Embedding lookup

Use tiny hand-computable inputs for unit tests. Include cases with large positive and negative logits to validate softmax stability.

Then implement the GPT-2 forward pass in this order:

1. Checkpoint header parsing and weight mapping
2. Token and learned position embeddings
3. First LayerNorm
4. Combined QKV projection
5. Head splitting and causal attention
6. Attention output projection and residual connection
7. Second LayerNorm
8. MLP expansion, GELU, projection, and residual connection
9. Repetition over all Transformer layers
10. Final LayerNorm
11. Vocabulary projection with tied embedding weights

Initially accept token IDs directly. Tokenization is deliberately deferred so it cannot obscure errors in the model.

Completion criteria:

- The full GPT-2 checkpoint loads with validated dimensions and file bounds.
- A forward pass completes for a short token sequence.
- No machine-learning or numerical library is required.
- All allocations and tensor layouts are explicit and documented.

## Phase 3: Achieve Numerical Parity

Add a diagnostic mode that can dump tensors from both the reference and C++ implementations.

Compare values in this order:

1. Input token IDs
2. Token and position embeddings
3. LayerNorm outputs
4. Q, K, and V projections
5. Attention scores before masking
6. Masked attention probabilities
7. Attention output
8. Residual stream after every block
9. Final normalized activations
10. Vocabulary logits

For every comparison, report at least:

- Maximum absolute error
- Maximum relative error
- Index and values at the largest discrepancy
- Whether all values are finite

Start with one thread, FP32, deterministic inputs, and no fast-math compiler options. Exact bitwise equality is not required because accumulation order can differ, but errors should remain small and explainable.

Generated text is not a sufficient correctness test. A small logit difference can select a different token, after which the two generations are no longer directly comparable.

Completion criteria:

- Intermediate activations agree with the reference layer by layer.
- Final logits remain within a documented tolerance.
- Greedy next-token selection matches across a collection of fixed prompts.

## Phase 4: Implement Tokenization and Generation

Implement GPT-2's byte-level BPE tokenizer as an independent module:

- UTF-8 input handling
- Byte-to-symbol mapping
- Pretokenization
- BPE merge ranks
- Encoding text to token IDs
- Decoding token IDs to bytes and text
- End-of-text token handling

Test the tokenizer separately against known token sequences before connecting it to the model.

Add generation strategies in this order:

1. Greedy decoding
2. Temperature scaling
3. Top-k sampling
4. Seeded random sampling
5. Repetition or frequency penalties as optional experiments

Keep logits available before sampling so generation policies never become entangled with the Transformer implementation.

Completion criteria:

- Known strings produce the expected GPT-2 token IDs.
- Encoding followed by decoding preserves representative UTF-8 input.
- Greedy generation matches the reference for fixed prompts.
- Sampling is repeatable when the seed and configuration are fixed.

## Phase 5: Add KV Caching

The initial implementation may recompute the entire prefix for every generated token. Preserve that path as the reference, then add a key-value cache for autoregressive decoding.

For every layer, store the keys and values produced by previous tokens. When generating a new token:

1. Compute Q, K, and V only for the new position.
2. Append the new K and V to the layer cache.
3. Attend with the new query over all cached positions.
4. Produce logits for the newest position only.

Test cache reset, context exhaustion, and prompts of different lengths.

Completion criteria:

- Cached and uncached inference produce matching logits at every generation step.
- Cache memory use is predictable from layer, context, head, and channel dimensions.
- Prefill and single-token decoding are measured separately.

## Phase 6: Introduce the Vulkan Backend

Keep the CPU backend as the permanent correctness oracle. Design the model representation so both backends consume the same configuration and weights.

```text
shared checkpoint and model description
              |
              +-- scalar CPU backend
              |
              +-- Vulkan compute backend
```

Port operations incrementally:

1. Matrix multiplication
2. Embedding lookup
3. LayerNorm reductions
4. QKV projection
5. Attention score calculation
6. Causal masking and softmax
7. Attention-value multiplication
8. GELU, projections, and residual operations

After each port, read back representative results and compare them with the CPU backend. Begin with separate, easily inspected dispatches. Shader fusion should come only after the complete Vulkan path is correct.

Topics to investigate:

- Buffer layout and alignment
- Weight upload and persistent device residency
- Workgroup dimensions
- Tiled matrix multiplication
- Shared-memory reuse
- Reduction algorithms
- Synchronization and pipeline barriers
- FP32 versus FP16 behavior
- Prefill throughput versus decoding latency

Completion criteria:

- Every Vulkan kernel has a CPU parity test.
- Full-model Vulkan logits match CPU logits within a documented tolerance.
- Weight upload is excluded from steady-state token-generation measurements.

## Phase 7: Optimize from Measurements

Profile before changing algorithms. Record separate measurements for checkpoint loading, tokenization, prefill, and per-token decoding.

Optimization order:

1. Remove unnecessary allocation and copying.
2. Improve CPU and GPU memory layouts.
3. Tile matrix operations.
4. Add CPU multithreading and SIMD as comparison points.
5. Fuse GPU operations where memory traffic justifies it.
6. Improve KV-cache layout and access patterns.
7. Add FP16 inference.
8. Experiment with INT8 and INT4 weight quantization.

For every optimization, require:

- Matching correctness tests
- Before-and-after timings
- Memory-use measurements
- A written explanation of the expected bottleneck

An important experiment is comparing prefill with decoding. Prefill presents large, parallel matrix operations, while single-token decoding often becomes dominated by repeatedly reading model weights.

## Phase 8: Study a Modern Llama-Style Architecture

After GPT-2 is complete, use a small model compatible with [`llama2.c`](https://github.com/karpathy/llama2.c) to study modern architectural changes without simultaneously debugging the fundamentals.

Implement and compare:

| GPT-2 component | Modern alternative |
| --- | --- |
| LayerNorm | RMSNorm |
| Learned position embeddings | RoPE |
| GELU MLP | SwiGLU |
| Multi-head attention | Grouped-query or multi-query attention |
| FP32 weights | Quantized weights |

This phase distinguishes the essential decoder-only Transformer algorithm from choices specific to GPT-2.

## Phase 9: Investigate Model Behavior

Add analysis tools rather than treating generated text as the only observable output:

- Visualize attention probabilities by layer and head.
- Track residual-stream norms through the network.
- Measure next-token entropy.
- List the highest-logit tokens at each generation step.
- Find nearest tokens in embedding space.
- Disable individual attention heads or MLP blocks.
- Measure logit changes after editing one prompt token.
- Compare early-layer and late-layer representations.

These experiments connect the implementation to questions about how information moves through the network.

## Optional Phase: Training

Training is valuable, but it should not block the inference project. Begin only after inference is numerically reliable.

A sensible progression is:

1. Manually derive and implement backward passes for the numerical primitives.
2. Train a tiny character-level model.
3. Add AdamW and gradient accumulation.
4. Verify gradients with finite differences.
5. Train a tiny Transformer rather than GPT-2 124M.

Training a useful large model is not necessary for understanding how trained weights participate in inference.

## Suggested Schedule

The estimates assume prior experience with C++, GPU compute, and mathematical implementation.

| Stage | Focused time |
| --- | ---: |
| Reference setup | 2-4 hours |
| CPU primitives and GPT-2 forward pass | 20-35 hours |
| Numerical parity and debugging | 8-15 hours |
| Tokenizer and generation | 10-20 hours |
| KV caching | 6-12 hours |
| Initial Vulkan backend | 25-50 hours |
| Optimization and experiments | Open-ended |

## Deliberate Non-Goals for the First Version

- GPT-3-scale execution
- Training GPT-2
- A general tensor or autograd framework
- CUDA or multiple GPU backends
- GGUF support
- Quantization
- Mixed precision
- FlashAttention
- Highly fused shaders
- Production serving features

Each may be worthwhile later, but none is required to understand and reproduce the core inference algorithm.

## Primary Resources

- [3Blue1Brown: Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt/)
- [3Blue1Brown: Attention in transformers, step by step](https://www.3blue1brown.com/lessons/attention/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [OpenAI GPT-2 repository](https://github.com/openai/gpt-2)
- [Karpathy: Neural Networks Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)
- [Karpathy: build-nanogpt](https://github.com/karpathy/build-nanogpt)
- [Karpathy: llm.c](https://github.com/karpathy/llm.c)
- [Karpathy: llama2.c](https://github.com/karpathy/llama2.c)

## Final Success Criteria

The project has reached its primary goal when:

- The C++ implementation loads an existing GPT-2 124M checkpoint.
- Its intermediate activations and logits match a trusted FP32 reference.
- It encodes prompts and generates text without Python or an ML framework.
- Cached and uncached inference agree.
- The CPU implementation remains available as a simple correctness oracle.
- The Vulkan implementation matches the CPU result and has measured performance characteristics.
- Architectural experiments can be conducted without modifying unrelated subsystems.
