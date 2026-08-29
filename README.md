# SansanE

SansanE is a learning project for implementing decoder-only Transformer inference from scratch in C++. The first target is GPT-2 124M, using a simple FP32 CPU implementation for correctness before adding tokenization, KV caching, Vulkan compute, and optimization.

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
