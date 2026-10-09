# Mesh GPT

A character-level autoregressive Transformer using CUDA Oxide. The decoder uses
multi-head causal exclusive self attention (XSA), QK-Norm, slice RoPE, learned token
embeddings, pre-RMSNorm, residual connections, a final RMSNorm, and a tied
vocabulary head. All mesh layers, normalization layers, and the vocabulary head
are bias-free; every RMSNorm is parameter-free. This follows
the decoder design described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
and the pre-norm placement in the [GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

The experimental change is to use sparse two-parent meshes for **all Q/K/V
projections, the attention output projection, and the feed-forward branch**.
Every mesh layer preserves channel width. Neuron `c` reads channels `c` and
`(c + 1) mod width` from the preceding layer, with two separate weights:

```text
z[c] = w_self[c] * x[c] + w_neighbor[c] * x[(c + 1) mod width]
```

Intermediate mesh layers use ELU with alpha 1; the last layer is linear so the
branch can produce signed Q/K/V vectors and residual updates. This extends the
two-parent idea of the scalar diamond mesh to vector inputs and outputs.
Mesh weights are independent across depth, projections, and Transformer blocks.

### Pre-RMSNorm

Each token is normalized across its full channel width with
[RMSNorm](https://arxiv.org/abs/1910.07467):

```text
inverse_rms = 1 / sqrt(mean(x * x) + 1e-5)
normalized = x * inverse_rms
out = normalized
```

Normalization has no learned scale, bias, or mean subtraction. The decoder
uses **pre-norm**: normalization precedes the attention and feed-forward
branches, while each residual adds to the
unnormalized branch input. A final RMSNorm precedes the tied vocabulary head.

```text
x = x + attention_branch(RMSNorm(x))
x = x + feed_forward_mesh(RMSNorm(x))
```

For upstream gradient `g`, CUDA uses
`dx = inverse_rms * (g - normalized * mean(g * normalized))`.
This includes the denominator derivative and remains finite for zero or tiny
inputs. Parameter-free normalization still propagates gradients to its input.

### QK-Norm

Every block applies RMSNorm separately to each projected Q/K head before RoPE,
following the per-head placement in
[Hugging Face's Qwen3 attention](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py).
The implementation here uses fixed normalization with epsilon `1e-5`, without
an affine scale or a learned attention temperature:

```text
q[h] = q[h] / sqrt(mean(q[h] * q[h]) + 1e-5)
k[h] = k[h] / sqrt(mean(k[h] * k[h]) + 1e-5)
q, k = slice_rope(q, k, position)
scores[h] = q[h] @ k[h].T / sqrt(head_width)
```

Means are taken over the channels within one head and one token. Q and K use
their own RMS values; heads and batch items cannot affect one another's
normalization. V keeps the output of its mesh. The same CUDA forward/backward
kernels implement full-width residual RMSNorm and per-head QK-Norm, storing one
inverse RMS per normalized group. Every block always uses QK-Norm.

### Slice RoPE

Each block rotates the projected queries and keys with the split-half convention
from [Hugging Face's Llama implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).
For a head of even width `d`, pair `i` joins channels `i` and `i + d/2`:

```text
frequency[i] = 10000^(-2 * i / d)
angle[i] = position * frequency[i]
cos = concat(cos(angle), cos(angle))
sin = concat(sin(angle), sin(angle))
x1, x2 = x[:d/2], x[d/2:]
rotated = concat(-x2, x1)
out = x * cos + rotated * sin
```

The frequencies use the base from [RoFormer](https://arxiv.org/abs/2104.09864).
Positions start at zero within each sequence in the batch. The embedding table
contains token embeddings; RoPE supplies position information through Q/K.
The CUDA kernel computes the angles, sine, and cosine. Its backward applies the
transpose rotation, equivalent to negating the angle. V goes directly to
attention and XSA. Every block always uses full-head slice RoPE with base 10000.

### Exclusive self attention

Attention first combines token positions using `softmax(Q K^T / sqrt(head_width)) V`;
only keys at or before the current position participate, including the current
token. Before the output mesh, each head applies
[Exclusive Self Attention](https://arxiv.org/html/2603.09078v1#S3):

```text
y[i] = sum_j attention[i, j] * v[j]
u[i] = v[i] / max(norm(v[i]), 1e-12)
z[i] = y[i] - dot(y[i], u[i]) * u[i]
```

Normalization is over channels within each head, separately for each token and
batch item. Above the norm floor, `z[i]` is orthogonal to the current token's
value vector. The floor matches
[PyTorch's F.normalize](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)
in the paper's algorithm; zero values leave `y[i]` unchanged. XSA adds no
parameters and always runs in every block. The vocabulary logits use dot
products with the shared token embedding table, with no additive bias.

```mermaid
flowchart LR
    T[Token IDs] --> E[Token embeddings]
    E --> N1[Fixed RMSNorm]
    N1 --> QKV[Q / K / V meshes]
    QKV --> NQK[Per-head QK-Norm on Q/K]
    NQK --> P[Slice RoPE on Q/K]
    P --> A[Causal multi-head attention]
    QKV -->|V| A
    A --> X[Per-head XSA projection]
    QKV -->|V| X
    X --> O[Output mesh]
    E --> R1[Residual add]
    O --> R1
    R1 --> N2[Fixed RMSNorm]
    N2 --> F[Feed-forward mesh]
    R1 --> R2[Residual add]
    F --> R2
    R2 --> Next[Next block / final fixed RMSNorm]
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
the tokenizer has 26 characters and the model has **3648 parameters**:
832 embedding weights and 2816 mesh edge weights. RMSNorm and QK-Norm add no
parameters.
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

On an RTX 3050 Laptop GPU, a 1000-step bias-free fixed RMSNorm + QK-Norm +
slice RoPE + XSA run with the default configuration produced:

```text
step      0: train CE=3.53211, validation CE=3.49710, perplexity=33.020
step   1000: train CE=1.33697, validation CE=1.71007, perplexity=5.529
Training time: 7.966 s
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
The `MSHGPT06` format identifies bias-free mesh projections, parameter-free
RMSNorm/QK-Norm, slice RoPE, and XSA. Older `MSHGPT01` and `MSHGPT02` checkpoints
used learned position embeddings; `MSHGPT03` used RoPE with LayerNorm, and
`MSHGPT04` still had mesh and vocabulary
biases; `MSHGPT05` used learned RMSNorm scales without QK-Norm. They are rejected
with a retraining message, so their weights cannot silently run under a
different architecture or parameter layout.

## GPU execution and verification

CUDA performs embedding lookup, RMSNorm/QK-Norm, mesh projections, RoPE and its
transpose rotation, attention and its softmax derivative, XSA projection and
its derivatives, cross-entropy, gradient aggregation, and AdamW. Parameter
gradients include both uses of the tied token embedding table. AdamW decay
applies to embedding and mesh edge weights. Host work covers tokenization,
choosing minibatches, reporting loss, checkpoint I/O, and sampling the next character.

Each token row owns its feature/cache writes; each parameter owns its gradient
reduction and update. Reverse attention gathers contributions from later queries
in the same sequence. Residual and Q/K/V gradients accumulate in distinct
buffers. Launch contracts validate lengths and scalar bounds. Private execution
plans establish descriptor relationships and vocabulary IDs are checked before
launch. Kernels use safe `DisjointSlice` access and no atomics or raw pointers.
There is one documented `unsafe` call to bind the crate's own embedded bundle.

XSA's backward pass differentiates both occurrences of the normalized self value
and its normalization denominator. The local self-value derivative is added to
the attention derivative from all causal queries. Below the norm floor, the
denominator is constant. The raw attention output is retained for this backward
pass; the XSA output feeds the output mesh.

Supported limits: width 2..64, 1..8 heads dividing width with even channels per
head, 1..4 blocks,
context 1..128, mesh depths 1..16, vocabulary 2..256, and at most 4096 token
rows per batch. Activations and gradient buffers use fixed-capacity row types;
the training workspace is reused between batches of the same shape. Inference
uses the same forward kernels with fresh buffers. Dropout and a KV cache are
left for later experiments.

`cargo test --lib --locked` checks configuration, tokenizer/window boundaries,
checkpoint round-trips and malformed input, strict causality/batch isolation,
parameter-free RMSNorm without mean subtraction and its input derivatives at
zero and tiny inputs, split-half RoPE pairs, norm preservation, relative position
identity, per-head QK-Norm isolation, per-head XSA orthogonality, and finite
differences for every parameter
in a tiny Transformer and both XSA inputs, including zero and tiny values.
Zero-input checks exercise mesh stacks and vocabulary logits with nonzero
weights to detect unwanted additive offsets.
GPU `verify` compares every logit and parameter gradient against f64 autodiff
at multiple widths, head counts, blocks, single-token sequences, and batch sizes
(including 129 rows).
It checks AdamW updates/reset, rejects malformed input, and trains a task whose
final-token prediction requires information from an earlier token.

The same `verify` command checks isolated CUDA XSA outputs and both input
derivatives against f64 autodiff at 129 rows, multiple head widths, and zero,
tiny, aligned, and ordinary values.
It also checks isolated CUDA RoPE forward, inverse, and backward against f64
autodiff for multiple head widths, positions through 127, sequence resets,
129-row batch tails, norm preservation, and rejected invalid launch arguments.
It checks isolated CUDA full-width RMSNorm and per-head QK-Norm outputs and
input derivatives against f64 autodiff for widths 2, 4, 6, 12, 32, and 64,
with head counts 1, 2, 4, and 8,
zero/tiny/constant inputs, 129-row batch tails, and rejected invalid launch
arguments, including oversized channel/head combinations and cache length
mismatches.
