# SansanE

SansanE is a learning project for implementing decoder-only Transformer inference from scratch in C++. The first target is GPT-2 124M, using a simple FP32 CPU implementation for correctness before adding KV caching, Vulkan compute, and optimization.

See [LLM_INFERENCE_LEARNING_PATH.md](LLM_INFERENCE_LEARNING_PATH.md) for the complete roadmap.

## Download the GPT-2 Reference Model

Run the following commands from the SansanE repository root:

```bash
cd ..
git clone https://github.com/karpathy/llm.c.git llm.c-reference
cd llm.c-reference
chmod +x dev/download_starter_pack.sh
./dev/download_starter_pack.sh
```

Karpathy's starter pack is approximately 1 GB and downloads the prepared GPT-2 model, tokenizer, numerical test data, and a small tokenized dataset. It does not require Python.

After downloading, the important files are available at these paths relative to the SansanE repository root:

| Artifact | Relative path | Purpose |
| --- | --- | --- |
| FP32 weights | `../llm.c-reference/gpt2_124M.bin` | Initial checkpoint for the C++ implementation |
| BF16 weights | `../llm.c-reference/gpt2_124M_bf16.bin` | Later reduced-precision experiments |
| Debug state | `../llm.c-reference/gpt2_124M_debug_state.bin` | Numerical parity tests and intermediate reference values |
| Tokenizer | `../llm.c-reference/gpt2_tokenizer.bin` | GPT-2 token encoding and decoding |

Keep these generated artifacts outside the SansanE repository. They are large binary inputs and should not be committed to Git.

To verify the required FP32 files from the SansanE repository root:

```bash
ls -lh \
  ../llm.c-reference/gpt2_124M.bin \
  ../llm.c-reference/gpt2_124M_debug_state.bin \
  ../llm.c-reference/gpt2_tokenizer.bin
```

The download workflow and binary artifacts come from [Karpathy's `llm.c`](https://github.com/karpathy/llm.c).

## Build and Inspect the Checkpoint

Configure and build the C++ checkpoint inspector from the SansanE repository root:

```bash
cmake -S . -B build
cmake --build build
```

Run it with the sibling-directory checkpoint and tokenizer paths plus a quoted
ASCII prompt:

```bash
./build/checkpoint_inspector \
  ../llm.c-reference/gpt2_124M.bin \
  --tokenizer ../llm.c-reference/gpt2_tokenizer.bin \
  --prompt "Hello, world!"
```

Generate multiple tokens by adding `--generate`:

```bash
./build/checkpoint_inspector \
  ../llm.c-reference/gpt2_124M.bin \
  --tokenizer ../llm.c-reference/gpt2_tokenizer.bin \
  --generate 3 \
  --prompt "Hello"
```

The tokenizer implements GPT-2's pre-tokenization and BPE rules for ASCII
input without external dependencies. Unicode prompts are rejected explicitly
for now. Token IDs are decoded as raw GPT-2 token bytes, so generated and
complete text can be printed after inference.

The original numeric-token interface remains available:

```bash
./build/checkpoint_inspector ../llm.c-reference/gpt2_124M.bin 0 1
```

The positional arguments after the checkpoint are GPT-2 token IDs in context
order. For example, a one-token context is:

```bash
./build/checkpoint_inspector /path/to/gpt2_124M.bin 0
```

The inspector validates the checkpoint and token sequence, prints the 16
parameter tensors, executes an uncached causal forward pass across all supplied
tokens, and reports vocabulary logits and the greedy next-token ID for the last
context position.

Numeric mode also supports autoregressive generation:

```bash
./build/checkpoint_inspector ../llm.c-reference/gpt2_124M.bin --generate 3 0 1
```

At each generation step, the inspector selects the highest-logit token, appends
it to the context, and recomputes the complete prefix. This deliberately simple
implementation does not use a KV cache yet.
