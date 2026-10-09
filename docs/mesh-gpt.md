# Mesh GPT

A character-level autoregressive Transformer using CUDA Oxide. The decoder uses
multi-head causal attention, learned token/position embeddings, pre-LayerNorm,
residual connections, a final LayerNorm, and a tied vocabulary head. This follows
the decoder design described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
and the pre-norm placement in the [GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

The experimental change is to use sparse two-parent meshes for **all Q/K/V
projections, the attention output projection, and the feed-forward branch**.
Every mesh layer preserves channel width. Neuron `c` reads channels `c` and
`(c + 1) mod width` from the preceding layer, with separate weights and a bias:

```text
z[c] = w_self[c] * x[c] + w_neighbor[c] * x[(c + 1) mod width] + bias[c]
```

Intermediate mesh layers use ELU with alpha 1; the last layer is linear so the
branch can produce signed Q/K/V vectors and residual updates. This extends the
two-parent idea of the scalar diamond mesh to vector inputs and outputs.
Mesh weights are independent across depth, projections, and Transformer blocks.
Attention combines token positions using `softmax(Q K^T / sqrt(head_width)) V`;
only keys at or before the current position participate. The vocabulary logits
use dot products with the shared token embedding table plus a learned bias.

```mermaid
flowchart LR
    T[Token IDs] --> E[Token + position embeddings]
    E --> N1[LayerNorm]
    N1 --> QKV[Q / K / V meshes]
    QKV --> A[Causal multi-head attention]
    A --> O[Output mesh]
    E --> R1[Residual add]
    O --> R1
    R1 --> N2[LayerNorm]
    N2 --> F[Feed-forward mesh]
    R1 --> R2[Residual add]
    F --> R2
    R2 --> Next[Next block / final LayerNorm]
    Next --> H[Tied vocabulary head]
    H --> L[Next-token logits]
```

## Run

Use the root README's CUDA Oxide setup first. The original scalar mesh example
remains the default binary; select this example with `--bin mesh-gpt`.

```bash
# Train, save weights, and generate a sample.
cargo oxide run --bin mesh-gpt -- train

# Generate from the saved checkpoint, without retraining.
cargo oxide run --bin mesh-gpt -- generate \
  --checkpoint target/mesh-gpt/model.bin \
  --prompt "the little robot " --tokens 240 --temperature 0.7

# Use your own UTF-8 corpus and a wider model.
cargo oxide run --bin mesh-gpt -- train \
  --dataset data/your-corpus.txt --width 64 --heads 4 --layers 2 \
  --context 64 --steps 3000 --output target/mesh-gpt-custom

# Check every CUDA operator against the independent CPU reference.
cargo oxide run --bin mesh-gpt -- verify

# Plot learning curves; requires NumPy and Matplotlib.
python3 scripts/plot-gpt.py

# All dimensions, optimizer, sampling, and output options.
cargo oxide run --bin mesh-gpt -- --help
```

Defaults: 2 blocks, width 32, 4 heads, context 32, attention mesh depth 4,
feed-forward mesh depth 6, batch size 8, 1000 AdamW steps, learning rate 0.003,
weight decay 0.01, and seed 7. On the included original English demo corpus,
the tokenizer has 26 characters and the model has **6426 parameters**.
The tokenizer builds a sorted Unicode-character vocabulary from the corpus.
Prompts must use that vocabulary. Temperature 0 selects greedy decoding;
positive temperatures sample from the softmax distribution with a reproducible
seed. Generation uses a sliding context window and recomputes attention for
each new character.

The corpus is split contiguously into 90% training and 10% validation before
sampling windows. Inputs and next-token targets stay within their split; neither
windows nor targets cross the boundary. Validation uses fixed windows from seed
101, so each report evaluates the same held-out examples. Reported training
cross-entropy is the mean over steps since the preceding report; step 0 is a
single initial batch. Perplexity is `exp(mean cross_entropy)` per character.

On an RTX 3050 Laptop GPU, the default 1000-step run produced:

```text
step      0: train CE=3.45846, validation CE=3.40589, perplexity=30.141
step   1000: train CE=1.31634, validation CE=1.85909, perplexity=6.418
Training time: 8.486 s
```

This is a compact model for experimenting with mesh channel mixing. Samples
learn recurring words and sentence patterns, with spelling errors at this scale.
The character-level metric is specific to this corpus and tokenizer.

## Files and APIs

- `src/gpt/config.rs`: checked dimensions, parameter groups, and initialization.
- `src/gpt/gpu.rs`: typed CUDA kernels, execution plan, workspace reuse, and
  the safe `GpuMeshGpt` API.
- `src/gpt/reference.rs`: separate f64 scalar autodiff for numerical validation.
- `src/gpt/text.rs`: character tokenizer and split-aware next-token windows.
- `src/gpt/checkpoint.rs`: versioned little-endian checkpoint format.
- `src/bin/mesh-gpt.rs`: training, generation, artifact export, and GPU verification.
- `data/mesh-gpt-demo.txt`: original small English corpus.

`GpuMeshGpt::logits` accepts a flat batch of equal-length token sequences.
`cross_entropy` evaluates next-token loss. `train_batch` performs one AdamW
update; repeated calls retain moments and the optimizer step count.
`loss_and_gradient` copies diagnostic gradients to the host. `parameters` and
`set_parameters` export/replace weights; replacement resets AdamW state.
`generate` decodes greedily or samples at the requested temperature.

```rust
use grokking_abmod::gpt::{GptConfig, GpuMeshGpt};

let config = GptConfig { vocab_size: 5, context: 3, ..GptConfig::default() };
let mut model = GpuMeshGpt::new(0, config, 7)?;
let inputs = [0, 1, 2, 4, 1, 2];
let targets = [1, 2, 3, 1, 2, 0];
let loss = model.train_batch(&inputs, &targets, 3, 0.003, 0.01)?;
let logits = model.logits(&inputs, 3)?;
```

`target/mesh-gpt/` holds `model.bin`, `loss.csv`, `sample.txt`, and `run.txt`.
The plot script adds `gpt-training.svg` and `gpt-training.png`.
Checkpoints contain configuration, the exact tokenizer, and weights. Loading
for generation preserves predictions; optimizer moments are reset when weights
are installed. Checkpoint readers validate dimensions before allocating, reject
invalid characters, nonfinite weights, truncation, and trailing bytes.

## GPU execution and verification

CUDA performs embedding lookup, LayerNorm, mesh projections, attention and its
softmax derivative, cross-entropy, all parameter/input derivatives, gradient
aggregation, and AdamW. Parameter gradients include both uses of the tied token
embedding table. AdamW decay applies to all parameters, including biases and
normalization parameters. Host work covers tokenization, choosing minibatches,
reporting loss, checkpoint I/O, and sampling the next character.

Each token row owns its feature/cache writes; each parameter owns its gradient
reduction and update. Reverse attention gathers contributions from later queries
in the same sequence. Residual and Q/K/V gradients accumulate in distinct
buffers. Launch contracts validate lengths and scalar bounds. Private execution
plans establish descriptor relationships and vocabulary IDs are checked before
launch. Kernels use safe `DisjointSlice` access and no atomics or raw pointers.
There is one documented `unsafe` call to bind the crate's own embedded bundle.

Supported limits: width 2..64, 1..8 heads dividing width, 1..4 blocks,
context 1..128, mesh depths 1..16, vocabulary 2..256, and at most 4096 token
rows per batch. Activations and gradient buffers use fixed-capacity row types;
the training workspace is reused between batches of the same shape. Inference
uses the same forward kernels with fresh buffers. Dropout and a KV cache are
left for later experiments.

`cargo test --lib --locked` checks configuration, tokenizer/window boundaries,
checkpoint round-trips and malformed input, strict causality/batch isolation,
and finite differences for every parameter in a tiny Transformer.
GPU `verify` compares every logit and parameter gradient against f64 autodiff
at multiple widths, head counts, blocks, and batch sizes (including 129 rows).
It checks AdamW updates/reset, rejects malformed input, and trains a task whose
final-token prediction requires information from an earlier token.
