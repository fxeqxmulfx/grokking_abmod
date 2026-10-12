# Neighbor residual mesh: five-seed plateau verification

The Square Neighbor residual mesh **W128, B2, M3** has a short memorization
interval on all five tested seeds. Three seeds (8, 10, and 11) satisfy the
25-epoch train ≥99% / test ≤10% plateau followed by a ≤250-epoch test rise.
All five reach test ≥99%. For a fixed-seed demonstration with train =100%,
**seed 11** is the best qualifying result: a 34-epoch perfect-train interval,
a 241-epoch rise, and test ≥99% at epoch 539.

## Protocol and results

Training uses Rust/CUDA from fresh initialization, division modulo 97,
one-hot operands, 4,656 equations per split, and a complete shuffled training
epoch: nine batches of 512 and a tail of 48. AdamW uses learning rate 0.001,
weight decay 0.1, betas (0.9, 0.98), epsilon 1e-8, and ten warmup updates.
All parameters, including sparse edges, use the same learning rate. Each
seed changes both initialization and the train/test split.

Both complete splits are evaluated **every epoch**. A run ends at the first
recorded test ≥99%, with a cap of 10,000 epochs. The relaxed acceptance rule
requires at least 25 contiguous epochs at train ≥99% / test ≤10%, then test
≥90% within 250 epochs of the last test ≤10% observation, followed by ≥99%.
Lengths are endpoint differences, rather than counts of observations.

| Seed | Train ≥99% / test ≤10% interval | Length | Train =100% / test ≤10% interval | Length | First test ≥90% | ≤10% → ≥90% rise | First test ≥99% | Relaxed match |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 7 | 91–156 | 65 | 119–156 | 37 | 420 | 264 | 588 | No: rise >250 |
| 8 | 79–107 | 28 | 103–107 | 4 | 287 | 180 | 414 | Yes |
| 9 | 95–156 | 61 | 129–156 | 27 | 434 | 278 | 586 | No: rise >250 |
| 10 | 85–115 | 30 | Epoch 115 only | 0 | 323 | 208 | 466 | Yes |
| 11 | 96–158 | 62 | 124–158 | 34 | 399 | 241 | 539 | Yes |

With the stricter **train =100%** plateau threshold, only seed 11 meets
both the 25-epoch minimum and the 250-epoch rise limit. Seeds 7 and 9 have
longer perfect-train intervals, but slower rises; seeds 8 and 10 have shorter
perfect-train intervals. This is five-seed evidence for a short delay in
generalization, rather than five replications of a long zero-error plateau.

The [five-seed accuracy graph](grokking-results/neighbor-plateau/short-mesh-seeds.png)
shows solid test and dashed train curves. The
[seed 11 accuracy and loss graph](grokking-results/neighbor-plateau/neighbor-mesh-short-plateau.png)
shows the complete 539-epoch run. During its perfect-train interval, test
accuracy rises from about 5% to about 10%; the plateau refers to the low-test
threshold, not a literally constant accuracy or loss. The subsequent rise
is smooth. It is suitable for a compact delayed-generalization demonstration,
but is slower than the 113–141-epoch ReLU butterfly transitions in the
[earlier five-seed study](short-mesh-results.md).

## Architecture and visualization

The model has **38,784 parameters**: a 194×128 input projection (24,832),
six 128-channel mesh layers with two edges per channel (1,536), and a
128×97 readout (12,416). There are two residual blocks, each containing:

```text
RMSNorm → linear Neighbor mesh → Square → linear Neighbor mesh → Square → linear Neighbor mesh
       └──────────────────────────── add the block input ──────────────────────────────┘
```

Each mesh output channel `c` reads channels `c` and `(c + 1) % 128`.
RMSNorm precedes each branch and the final readout, with no learned affine
parameters. There are no biases. Mesh layers initialize as identity; dense
matrices use fan-in initialization.

For visualization, select one block output channel and show its branch
ancestors as **4 → 3 → 2 → 1** nodes across three mesh layers: ten nodes
and twelve learned edges, plus the skip connection. Adjacent output paths
share intermediate nodes. The displayed inputs are values after RMSNorm;
its scale uses all 128 channels and should be shown as a separate operation.
Dense input/readout matrices can stay as heatmaps. This is a view of the
actual model, with no pruning or training changes.

## Replay and numerical verification

Seed 7 replays the saved 25-epoch-cadence run with the same initialization.
The initial checkpoint hashes are identical, and **all 24 common observations**
(epochs 0 through 575) match exactly at CSV precision for loss, accuracy,
weight norm, learning rate, and training progress. Elapsed time is excluded.
The original coarse estimate of 25 perfect-train plateau epochs becomes
37 epochs at the finer cadence; the 275-epoch rise becomes 264, and the
600-epoch ≥99% observation becomes 588. See the
[replay check](grokking-results/neighbor-plateau/replay-check.json).

The CUDA verification suite includes this exact W128/B2/M3/Square/Neighbor
configuration. It compares logits, loss, and every parameter gradient
against independent f64 automatic differentiation, using gathered batches
of 1, 37, and 130 equations with repeated indices. It also checks three
AdamW updates, offset minibatches, reset repeatability, and validation.
The CPU central-difference gradient test now includes Neighbor residual
branches with both Square and ReLU.
Both checks passed. The exact candidate's maximum logit and gradient errors,
normalized by each array's maximum reference magnitude, were approximately
1.0e-6 and 1.1e-6. The complete
[CUDA verification log](grokking-results/neighbor-plateau/cuda-verification.log)
records the candidate and the other supported configurations.
[Verification provenance](grokking-results/neighbor-plateau/verification.json)
records the checked executable/source hashes, passing commands, and the local
NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB).

Complete curves, exact argument arrays, run metadata, classifications, and
initial/final checkpoint hashes are in the
[archived summary](grokking-results/neighbor-plateau/summary.json) and its
`raw/` directory. The copied coarse reference is separate from the five
replications. NVIDIA runs were sequential, with the continuous search
stopped during verification.

## Reproduce

```bash
cargo oxide build --arch sm_86 -- --release --locked --bin mesh-gpt
target/release/mesh-gpt grok-simple --verify
cargo test --locked --lib reference_gradients_match_central_differences
cargo build --release --locked --bin sharp-mesh-search

# Stop other training jobs before launching these five sequential runs.
target/release/sharp-mesh-search \
  --output target/grok-div97-search/neighbor-plateau-check \
  --trainer target/grok-div97-search/sharp-mesh-loop/bin/mesh-gpt \
  --plateau-epochs 25 --report-every 10 \
  --queue experiments/div97-neighbor-plateau.args --max-trials 5

UV_CACHE_DIR=/tmp/grokking-uv-cache uv run --locked scripts/analyze-short-mesh.py \
  target/grok-div97-search/neighbor-plateau-check \
  --output docs/grokking-results/neighbor-plateau --detail-seed 11 \
  --replay-reference target/grok-div97-search/sharp-mesh-loop/trials/000021-residual-mesh-mlp-w128-d2-s7-a01
```

The completed runs use the same isolated CUDA trainer as the earlier studies.
The later verification build adds a fixture and descriptive verification
labels; it changes no neural kernels or training behavior. The table above
records Rust/CUDA measurements. The [browser implementation](../web/own-channel/README.md)
previously used this model as its second tab with seed 11, handwritten WASM
scalar/SIMD and WebGPU, and every-epoch measurements. Both browser engines
retain the 34-epoch perfect-train interval: WASM SIMD reaches test ≥99% at
epoch 532, and WebGPU at 577. Their rises take 238 and 243 epochs respectively.
See [browser measurements](grokking-browser-models.md) for the complete comparison.
Both current tabs use the [shared-E Rotate90 variant](rotary-neighbor-results.md),
with the same residual branches and widths 128 and 1024.
