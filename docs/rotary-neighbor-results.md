# Shared-E Rotate90 Neighbor measurements

The two local browser tabs use the same Square Neighbor residual architecture
at widths **128** and **1024**. Only width changes: the compact model reaches
held-out accuracy ≥99% without a memorization plateau, while the wider model
meets the short-plateau grokking criterion on **all five tested seeds**.
The [browser implementation](../web/own-channel/README.md) trains from scratch
with live weights, activations, loss, and accuracy.

## Architecture

One learned table `E` of shape `97 × W` supplies both operand embeddings and
the output readout. A fixed +90° rotation distinguishes operand `b`:

```text
J(u, v) = (-v, u)                         # each adjacent channel pair
x0 = E[a] + J(E[b])

for block in 0..2:
    r = RMSNorm(x)
    z1 = M1(r)
    z2 = M2(z1²)
    branch = M3(z2²)
    x = x + branch

logits = RMSNorm(x) Eᵀ
M(z)[c] = α[c] z[c] + β[c] z[(c + 1) mod W]
```

Each block has its own three mesh layers; every layer has independent signed
self and neighbor weights. RMSNorm uses all channels, epsilon 1e-5, and no
learned affine parameters. All layers are bias-free. The six mesh layers
contribute `12W` parameters and `E` contributes `97W`, giving **109W** unique
parameters: **13,952** at W128 and **111,616** at W1024.

The role rotation is inspired by the pairwise rotations in
[RoFormer §3.2](https://arxiv.org/html/2104.09864v5). Its angle is fixed by
operand role, independent of residue value. The model has no attention or
position-dependent frequency ladder. Backpropagation adds the `a` lookup,
`Jᵀ`-rotated `b` lookup, and readout gradients into one gradient for `E`, with
one AdamW state per unique parameter.

SplitMix64 initializes `E` uniformly in `[-1/√97, +1/√97]`. Each mesh starts
with `α=1, β=0`; an initial branch therefore computes `RMSNorm(x)⁴`.
Its skip adds that branch to `x`.

For visualization, a selected output channel's branch has **4 → 3 → 2 → 1**
ancestors: ten nodes, twelve learned edges, and the residual skip. The local
diagram displays actual probe activations. Its normalization depends on all
W channels, including channels outside the displayed ancestor tree.

## Protocol and criterion

Division modulo 97 has 9,312 ordered pairs, with `a ∈ [0,96]` and
`b ∈ [1,96]`. Each seed fixes initialization and a shuffled 50% split:
4,656 training pairs and 4,656 held-out pairs. Every epoch visits all training
pairs once using nine 512-row batches and a 48-row tail. Both complete splits
are evaluated every ten updates, exactly once per epoch.

All reported runs use two blocks, three Neighbor layers per branch, Square,
AdamW learning rate 0.001, decay 0.1, betas (0.9, 0.98), epsilon 1e-8, and ten
warmup updates. They stop at the first held-out accuracy ≥99% or 10,000 epochs.

A qualifying curve needs at least 25 contiguous epochs with train ≥99% and
held-out ≤10%, then a rise of at most 250 epochs from the last ≤10% report to
the first ≥90%, followed by ≥99%. Train dips and missing epoch observations
break the interval. The train =100% interval is recorded separately. Interval
length is `end − start`; a 34-epoch interval contains 35 endpoint-inclusive
observations. The measured held-out rises are continuous over dozens of epochs.

## Five-seed Rust/CUDA replication

Runs were sequential on an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GiB,
driver 615.71.09, and CUDA 13.4. Other training jobs were stopped during these
measurements. W128 reaches the target on every seed, with **0/5** qualifying
memorization plateaus:

| Seed | First train ≥99% | Test at that report | Last test ≤10% | First test ≥90% | Rise (epochs) | First test ≥99% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 105 | 88.79% | 60 | 107 | 47 | 131 |
| 8 | 102 | 87.59% | 50 | 106 | 56 | 133 |
| 9 | 90 | 87.91% | 48 | 94 | 46 | 123 |
| 10 | 104 | 86.47% | 57 | 109 | 52 | 137 |
| 11 | 101 | 85.72% | 57 | 106 | 49 | 133 |

W1024 reaches the target and qualifies on **5/5** seeds. Each also has at least
25 epochs of perfect training accuracy while held-out accuracy remains ≤10%:

| Seed | Train ≥99% plateau | Length | Train =100% plateau | Length | First test ≥90% | Rise | First test ≥99% |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 7 | 15–57 | 42 | 22–57 | 35 | 135 | 78 | 187 |
| 8 | 15–57 | 42 | 22–57 | 35 | 133 | 76 | 175 |
| 9 | 15–57 | 42 | 24–57 | 33 | 136 | 79 | 190 |
| 10 | 16–57 | 41 | 23–57 | 34 | 131 | 74 | 176 |
| 11 | 15–56 | 41 | 22–56 | 34 | 136 | 80 | 187 |

![Five-seed W1024 loss and accuracy](grokking-results/rotary-neighbor-wide/short-mesh-seeds.png)

An eight-run seed-11 screening varied width, block count, branch depth, and
decay. W1024 B2 M3 was the only qualifying configuration. W512 with decay 0.1
had a three-epoch ≥99%-train plateau, below the criterion. This screening
motivated the five-seed replication; its other configurations are recorded in
[the variant archive](grokking-results/rotary-neighbor-variants/summary.json).

## Fresh browser runs

The browser defaults to seed 11. All engines start from the same byte-identical
Rust initialization; complete metrics are actual training observations.

| Width / engine | Train ≥99% plateau | Train =100% plateau | First test ≥90% | Rise | First test ≥99% | Training + evaluation |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 128 / Rust CUDA | none | none | 106 | 49 | 133 | — |
| 128 / WASM SIMD | none | none | 106 | 49 | 133 | 9.32 s |
| 128 / hardware WebGPU | none | none | 106 | 49 | 133 | 1.73 s |
| 1024 / Rust CUDA | 15–56 (41) | 22–56 (34) | 136 | 80 | 187 | — |
| 1024 / WASM SIMD | 15–56 (41) | 22–56 (34) | 136 | 80 | 187 | 95.09 s |
| 1024 / hardware WebGPU | 15–56 (41) | 22–56 (34) | 135 | 79 | 186 | 13.40 s |

WASM measurements used one thread in Node v24.21.0 on an AMD Ryzen 5 5600H.
WebGPU used Chrome 155.0.8059.39 and a hardware adapter (`fallback: false`);
the adapter description is hidden. Timings exclude engine setup and normal
live visualization readbacks. They describe this machine and measurement
procedure. Different f32 reduction orders can move crossing epochs.

Fresh runs through the real UI worker also reached ≥99% at epochs 133 and 186.
The compact tab reported `Generalized`; the wide tab reported
`Short-plateau grokking`. Raw browser histories, CSV exports, source hashes,
measurement metadata, and UI outcomes are in
[`web/own-channel/tests/results/`](../web/own-channel/tests/results/).

## Verification and archived evidence

The independent f64 reference differentiates every parameter and checks central
finite differences, including the shared embedding's three roles, Square,
RMSNorm, skip, and circular neighbor edges. Rust host tests passed (56 library
tests and nine scheduler tests), as did strict Clippy and formatting.

The CUDA verifier passed 42 model configurations plus validation checks.
Exact W128 and W1024 fixtures check logits, loss, every gradient group for
1/37/130 gathered rows, offset AdamW updates, and bitwise repeatability. Maximum
gradient errors relative to each group's maximum were 7.5e-7 and 1.3e-6.
Typed launch contracts reject odd rotary widths before launch.

The complete Node suite passed 73 tests; a subsequently added checkpoint test
also passed separately for widths 128/1024 and seeds 7/11. Scalar and SIMD
checks cover batches 1/37/48/512, signed mesh weights, wraparound, and three
AdamW updates. WebGPU passed 18 gradient/optimizer checks per width against the
f64 reference. Both UI tabs were checked for independent state, controls,
live weight changes, channel selection, and viewport fit from 320×568 to
1920×1080, including tablet and landscape layouts.

Archives retain raw Rust CSVs, arguments, outcome metadata, source/checkpoint
hashes, and plots:

- [W128 five-seed study](grokking-results/rotary-neighbor/summary.json).
- [W1024 five-seed study](grokking-results/rotary-neighbor-wide/summary.json).
- [W1024 verification metadata](grokking-results/rotary-neighbor-wide/verification.json)
  and [CUDA verification log](grokking-results/rotary-neighbor-wide/cuda-verification.log).
- [Seed-11 replay check](grokking-results/rotary-neighbor-wide/replay-check.json):
  all saved numerical fields match at 188 common observations (epochs 0–187),
  with elapsed time excluded and comparison limited to CSV precision.
- [Earlier untied Neighbor results](neighbor-mesh-results.md) and
  [concat/dense/butterfly comparisons](grokking-browser-models.md).

## Reproduce

```bash
cargo oxide build --arch sm_86 -- --release --locked --bin mesh-gpt
target/release/mesh-gpt grok-simple --verify
cargo test --locked --lib
cargo test --locked --bin sharp-mesh-search
cargo clippy --locked --all-targets -- -D warnings
cargo build --release --locked --bin sharp-mesh-search

# Stop other training jobs before running the sequential replication.
target/release/sharp-mesh-search \
  --output target/grok-div97-search/rotary-neighbor-wide-check \
  --trainer target/release/mesh-gpt \
  --plateau-epochs 25 --report-every 10 \
  --queue experiments/div97-rotary-neighbor-wide.args --max-trials 5

UV_CACHE_DIR=/tmp/grokking-uv-cache uv run --locked scripts/analyze-short-mesh.py \
  target/grok-div97-search/rotary-neighbor-wide-check \
  --output docs/grokking-results/rotary-neighbor-wide --detail-seed 11

node web/own-channel/server.mjs
```

Use a new output directory for a fresh replication: the scheduler resumes
existing studies. The W128 manifest is
[`experiments/div97-rotary-neighbor-plateau.args`](../experiments/div97-rotary-neighbor-plateau.args).
The app and training remain local; Node serves static files on
`http://127.0.0.1:8080`.
