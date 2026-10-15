# Shared-embedding concat and delayed-grokking model measurements

The later [short-plateau study](short-mesh-results.md) relaxes the minimum
plateau to 25 epochs and identifies replicated sparse mesh candidates.
Both current browser tabs use the [Dense + ReLU architecture with Normal(0, 1/W) E](rotary-dense-embedding-results.md)
at widths 128 and 1024, with one pre-norm residual block and shared E.
The native seed-11 W1024 run has a 92-epoch memorization plateau and qualifies
under the short-study criterion. The [Neighbor study](rotary-neighbor-results.md)
retains its five-seed experiments and earlier browser measurements.
The earlier-model comparisons below retain the original longer-plateau rule.

## Earlier untied Neighbor browser model

The Square Neighbor residual mesh uses W128, two blocks with three mesh layers
each, 38,784 parameters, seed 11, and every-epoch complete-split evaluation.
Its initial raw f32 checkpoint is byte-identical to Rust:
`0e289f2ea6784e257cdc2acc60fb238d00886d04c402bc34846da009879a43ea`.
Each browser implementation starts from that initialization and trains locally.

| Implementation / seed 11 | Train =100% / test ≤10% | Test ≤10% → ≥90% | First test ≥99% | Training + evaluation time |
| --- | --- | ---: | ---: | ---: |
| Rust/CUDA | 124–158 (34 epochs) | 241 epochs | 539 | 15.78 s |
| Node WASM SIMD | 124–158 (34 epochs) | 238 epochs | 532 | 36.04 s |
| Chrome hardware WebGPU | 124–158 (34 epochs) | 243 epochs | 577 | 4.49 s |

The train ≥99% / test ≤10% plateau is 62 epochs in all three. The browser label
uses the short-study 25-epoch plateau / 250-epoch rise criterion. Both browser
runs satisfy it. Adapter description is hidden, and timings exclude engine
setup and normal live visualization readbacks. They are local observations.

The browser has a handwritten scalar/SIMD WASM trainer and deterministic
WebGPU kernels with only a 512-row activation cache. A selected branch channel
is shown as 4→3→2→1 ancestors plus a skip, and its weight atlas shows all
three layers' self/neighbor edges. Complete fresh browser runs and the f64
gradient/optimizer checks are in
[`web/own-channel/tests/results/`](../web/own-channel/tests/results/).

## Earlier comparison protocol

These studies use division modulo 97, a fixed half-data split, seed 7,
4,656 equations per split, batch 512 with an exact 48-equation epoch tail,
AdamW learning rate 0.001, weight decay 0.1, betas (0.9, 0.98), ten warmup
updates, and complete-split evaluation every 250 updates / 25 epochs.
Runs stop at the first observed held-out accuracy of at least 99%, or at
10,000 epochs. Variant-specific decay is indicated below.

## Shared embedding with ordered concatenation

The new `tied-concat-mesh-mlp` stores one embedding `E` of shape 97×64,
an ordered concat projection `A` of shape 128×64, and four independent
own-channel gain vectors. It computes
`ReLU([E[a], E[b]] A) * product_l ReLU(g_l)`, followed by the tied readout
`E^T`. Input roles and readout share one gradient and one AdamW state for `E`.
The two halves of `A` keep division operand order distinguishable.

The original own-channel model stores a 194×64 input matrix, a separate
64×97 readout, and the same four independent gain vectors. The paired Rust
runs both start from initialization and use the same training protocol.

| Model | Parameters | First train ≥99% | Test at that report | First test ≥99% | Train/test delay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared E + concat, own depth 5 | 14,656 | 425 | 23.58% | 1,300 | 875 |
| Original own depth 5 | 18,880 | 475 | 2.19% | 8,925 | 8,450 |

The concat model has 22.4% fewer unique parameters and reached the target in
6.87 times fewer updates. Its measured Rust training/evaluation duration was
16.14 seconds versus 76.62 seconds for the original, about 4.75 times less
total time. The shared projection makes each update more expensive; these
results do not establish a per-update speedup from fewer parameters.

Concat has a train/test delay, but no qualifying five-observation memorization
plateau: train was already accompanied by substantial held-out accuracy at
the first ≥99% train observation. Its sampled ≤10% → ≥90% transition took
650 epochs. It was the first browser tab and does not satisfy the stricter
plateau-plus-sharp-rise search criterion. Fifty epochs were insufficient.

Reproduce the paired study with
[`experiments/div97-tied-concat.json`](../experiments/div97-tied-concat.json):

```bash
cargo oxide build --arch sm_86 -- --release --locked --bin mesh-gpt
uv run --locked scripts/run-mesh-search.py experiments/div97-tied-concat.json
```

The complete normalized summaries, raw metrics, metadata, and model/source
hashes are in [`grokking-results/tied-concat/`](grokking-results/tied-concat/).
The seed-7 concat initial raw f32 weights have SHA256
`862a43552934966b36d8337b782701cf554c6ef52672c1f3a68ba8acb146e32b`;
the browser initialization matches byte for byte.

## Dense residual control and browser runs

The existing `residual-mlp` with **width 64, two residual blocks, expansion 4**
has 84,160 parameters. Each pre-RMSNorm branch uses dense 64×256 expansion,
ReLU, and 256×64 contraction, followed by a skip addition. Final RMSNorm
precedes the readout. A label containing `residual2-...-d64` in the old study
denotes width 64, not 64 residual blocks.

The Rust control first reached train ≥99% at epoch 50, with an early train
dip; the first confirmed plateau starts at epoch 150 and is confirmed at 250.
Held-out accuracy stayed near 2% for thousands of epochs. Its last sampled
test ≤10% observation was **6.68% at epoch 4,625**, followed by **93.54% at
5,225**, a **600-epoch** rise. The first test ≥99% was epoch 5,350.

The earlier browser benchmark implemented both models from scratch on WASM and WebGPU:

| Model / implementation | First train ≥99% | First test ≥99% | Sampled ≤10% → ≥90% rise | Final test accuracy | Training + evaluation time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Concat / Rust CUDA | 425 | 1,300 | 650 | 99.420% | 16.14 s |
| Concat / Node WASM SIMD | 425 | 1,275 | 650 | 99.270% | 24.93 s |
| Concat / Chrome WebGPU | 425 | 1,225 | 650 | 99.055% | 5.43 s |
| Dense residual / Rust CUDA | 50 | 5,350 | 600 | 99.485% | See the original study |
| Dense residual / Node WASM SIMD | 50 | 5,125 | 500 | 99.570% | 839.43 s |
| Dense residual / Chrome WebGPU | 50 | 5,125 | 475 | 99.334% | 83.33 s |

CPU measurements used Node v24.21.0 and one thread on an AMD Ryzen 5 5600H.
Chrome used a hardware WebGPU adapter; adapter description was not exposed.
Times exclude engine setup and do not include normal live visualization
readbacks. They are measurements on this environment, not portable timing
guarantees. Floating-point reductions and approximations differ between
engines; matching initialization and protocol does not imply matching curves.

The dense control's longest contiguous train ≥99% / test ≤10% intervals
before its first test ≥90% were 800 epochs on WASM and 575 on WebGPU:
occasional train dips interrupt an otherwise prolonged low-test phase.
Neither browser curve satisfies the stricter 1,000-contiguous-epoch /
250-epoch-rise rule. The dense control has since been replaced in the second
tab by the verified Neighbor mesh; its code and measurements remain as references.

The dense seed-7 initial raw f32 weights have SHA256
`24ca4103d5215dd0c4df4bf703c1a3a5bbbe37e43326ef00818747e2834a6248`;
the browser initialization matches the original Rust artifact byte for byte.

Unmodified browser histories and measurement metadata, CSV exports, and
recomputed classification analysis are in
[`web/own-channel/tests/results/`](../web/own-channel/tests/results/).
A timestamp-only classifier bug was corrected after measuring: the first
confirmed plateau now remains fixed across later qualifying streaks. The
original metadata is retained alongside corrected analysis. No training
losses, accuracies, gradients, updates, or timings were changed.

## Residual mesh comparison

`residual-mesh-mlp` replaces each dense expansion/contraction branch with a
width-preserving sparse mesh. Every branch begins with RMSNorm, uses a linear
first mesh layer and activated intermediate layers, and adds its final linear
mesh output to the block input. Final RMSNorm precedes the dense readout.
Own, neighbor, and butterfly connectivity are supported. Mesh initialization
is identity; dense input/readout initialization follows fan-in scaling.

`W`, `B`, and `M` below denote width, residual blocks, and mesh layers per
branch. All twelve new runs reached held-out ≥99%; none combines a long
memorization plateau with a ≤250-epoch rise.

| Mesh configuration | Parameters | First train ≥99% | First test ≥99% | Sampled ≤10% → ≥90% rise |
| --- | ---: | ---: | ---: | ---: |
| Butterfly W64 B2 M4 | 19,648 | 300 | 1,275 | 575 |
| Neighbor W64 B2 M4 | 19,648 | 325 | 1,800 | 925 |
| Butterfly W64 B2 M8 | 20,672 | 300 | 1,300 | 550 |
| Butterfly W64 B4 M4 | 20,672 | 300 | 1,400 | 650 |
| Butterfly W64 B2 M4, decay 1 | 19,648 | 175 | 225 | 75 |
| Butterfly W64 B2 M4, decay 0.5 | 19,648 | 225 | 375 | 100 |
| Butterfly W128 B2 M8 | 41,344 | 125 | 1,225 | 750 |
| Butterfly W256 B2 M4 | 78,592 | 75 | 600 | 300 |
| Butterfly W128 B2 M16 | 45,440 | 125 | 1,300 | 600 |
| Butterfly W128 B4 M8 | 45,440 | 125 | 1,275 | 750 |
| Butterfly W128 B2 M8, decay 0.03 | 41,344 | 125 | 2,050 | 1,450 |
| Butterfly W128 B2 M16, decay 0.03 | 45,440 | 125 | 3,675 | 1,900 |

Strong decay gives short transitions without the required preceding plateau.
Wider configurations can memorize before generalizing, but their rise remains
gradual. Some later trials overlap other GPU trials; their elapsed times
should not be compared as serial performance benchmarks.

The manifest is
[`experiments/div97-residual-mesh.json`](../experiments/div97-residual-mesh.json),
with raw curves, metadata, and hashes in
[`grokking-results/residual-mesh/`](grokking-results/residual-mesh/).

## Earlier mesh and Transformer evidence

The [five-seed Neighbor Square verification](neighbor-mesh-results.md) measures
W128 B2 M3 with every-epoch evaluation: all five reach test ≥99%, and three
meet the short 25-epoch/250-epoch plateau/rise rule. Seed 11 also has a
34-epoch train =100% / test ≤10% interval, then a 241-epoch rise. This architecture
previously occupied the second browser tab. Its archived browser measurements
remain above; the current two-width comparison uses
[shared embeddings with a fixed operand rotation](rotary-neighbor-results.md).

The saved neighbor mesh MLP W56 depth 9 has a confirmed memorization plateau,
but its last test ≤10% observation is epoch 875 and first test ≥90% is 3,900:
a **3,025-epoch** rise, with test ≥99% at 5,075. W64 depth 9 rises over 3,250
epochs and reaches the target at 5,850. Both are gradual transitions.

The saved table/unit-normalized mesh Transformer W128 H4 L2 A8 F16 can rise
from ≤10% to ≥90% over 50 epochs, but train and held-out accuracy grow together:
it has no preceding memorization plateau. The ordinary table variant first
reaches train ≥99% with test already around 81%, also without such a plateau.
Reviewing saved Transformer histories found no completed curve combining the
required plateau and sharp rise. Several normalized files duplicate the same
underlying run and are not additional independent experiments.

An older W1024 H8 L5 mesh pilot was retired at epoch 400 after showing a
qualifying short plateau; it was not trained for the full 10,000-epoch budget.
Its saved curve cannot exclude later grokking.

The new [continuous Rust search](sharp-mesh-search.md) evaluates fresh mesh
MLPs, residual meshes, and mesh Transformers to the full configured budget,
records sampled plateau and transition durations, and archives qualifying
models separately. Its default rule is at least 1,000 contiguous epochs with
train ≥99% / test ≤10%, then a ≤250-epoch rise to ≥90%, followed by ≥99%.
