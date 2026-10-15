# Archived Neighbor frontend documentation

Superseded by the Dense + ReLU implementation described in [README.md](../web/own-channel/README.md).
All architecture, performance and current-mode statements below describe the
previous Neighbor frontend at the time of this archive. Its results remain
available for comparison.

# Rotate90 Grokking Lab

**Current mode: local only.** Node serves static files on localhost; all
training runs in the browser. The previous Sites identity remains in ignored
`.openai/hosting.json`. Its cloud removal has not been confirmed. Do not
publish updates without a new user request.

Both tabs use the smallest tested branch: **one residual block, one Neighbor
Mesh, and Square after the Mesh**, with one shared embedding `E`. Only width
changes:

| Tab | Channels | Unique parameters | Five-seed Rust result |
| --- | ---: | ---: | --- |
| Rotate90 · 128 | 128 | 12,672 | Test ≥99% at epochs 105–113; no memorization plateau |
| Rotate90 · 1024 | 1024 | 101,376 | Test ≥99% at epochs 178–188; an 11–15 epoch plateau |

The wider model's plateau is below the unchanged 25-epoch grokking criterion.
Both default runs report `Generalized`. Earlier two-block, three-Mesh browser
measurements remain archived in [the Rust and browser study](rotary-neighbor-results.md).

Hardware WebGPU is preferred, with handwritten WASM SIMD128 and scalar WASM
fallbacks. No training backend, pretrained weights, Python runtime, or
JavaScript ML framework is required. Node v24.21.0 was used for checks;
`.nvmrc` selects it.

```sh
node web/own-channel/server.mjs
```

Open `http://127.0.0.1:8080`. The dashboard fits the viewport without page
scrolling. Dynamics, Training, Weights, and Network views are available at every
size. Dynamics opens by default. On short phones it keeps the main training
buttons and metrics visible; the full settings remain in Training.
Tabs retain independent weights, AdamW moments, sampler, and curves;
switching tabs pauses the previous experiment. Changing engine or seed resets
the selected model. Switching views preserves training state.

## Languages and the opening dialog

The interface supports English, Spanish, German, Japanese, French, and Russian.
The first five follow [W3Techs' website content-language ranking](https://w3techs.com/technologies/overview/content_language)
on 12 October 2026; Russian is included for the user's original explanation.
This is a ranking of website content, rather than counts of Internet users.

Automatic selection uses the browser's language preference order, including
regional tags such as `es-MX`, `fr-CA`, and `ru-KZ`, with an English fallback.
The header and opening dialog share a language selector. A manual choice is
stored locally; choosing Auto restores browser selection and follows later
browser language changes. `?lang=en` (or another supported code) temporarily
overrides selection for links and browser checks. Switching language redraws
labels, tooltips, accessibility text, and live statuses while preserving model
weights, AdamW state, and learning history. Machine-readable exports retain
their original fields and phase identifiers.

Every page load opens **Division modulo 97**. Four pages explain the task,
the inverse-based solution, eight checked examples, and the current Rotate90
model. It includes the worked `10 / 3 = 68` example and a rendered Mermaid
flowchart. The ∑ header button reopens it; closing the dialog leaves training
under manual control. Its model page describes the actual pair inputs and
97 answer classes. Its native SVG shows the two E lookups, role rotation,
the RMSNorm → Neighbor Mesh → Square branch, the identity skip, and the
tied readout. Dashed lines indicate reuse of the same E. The model diagram
uses a vertical layout on phones, with localized labels in both layouts.
The model page also has Types and Equations views: scalar domains, tensor
shapes, row-vector conventions, channel/pair/class index bounds, and explicit
quantifiers for Mesh, Square, tied readout, Rotate90 and RMSNorm. It distinguishes
the correct answer `q` from the prediction `q̂`; ties choose the smallest class.
No Transformer/EOS reference section is included.

Mermaid Tiny 12.1.0 is a pinned **development dependency**. The frontend loads
only the selected language catalog, a roughly 21 KB solution SVG and a roughly
7 KB model SVG, without a Mermaid runtime or diagram CDN. Editable Mermaid
sources and their hash
manifest are in [`diagrams/`](../web/own-channel/diagrams/manifest.json); rendered SVGs are in
`dist/diagrams/`. The static preview serves them as `image/svg+xml`.
Model SVGs are generated directly by `scripts/build-model-diagrams.mjs`.

```sh
# Rebuild translation catalogs after editing scripts/translations*.mjs.
cd web/own-channel
npm run build:locales

# Rebuild the native model SVGs without a browser or extra dependency.
npm run build:model-diagrams

# Re-render SVGs against an isolated Chrome profile with CDP enabled.
GROKKING_CDP_PORT=9225 npm run build:diagrams
```

The [localization browser check](../web/own-channel/tests/results/rotary-localization-verification.json)
covers automatic regional selection, persisted manual choices, browser language
changes, dialog startup/reopening, and language switches after real WASM
training. It checks all four dialog pages and all four workspace views at
1366×768, 390×844, 320×568, and 844×390 for every language. The existing
two-model smoke check separately exercises GPU/WASM training, zoom, complete
weight rendering, independent tab state, and layout. Catalog tests require
every translated message and preserve all interpolation fields; diagram tests
check the rendered SVGs against their Mermaid source hashes.

## Shared architecture

One `97 × W` table supplies both operands and the tied readout. A fixed +90°
rotation marks operand `b` in each adjacent channel pair:

```text
J(u, v) = (-v, u)
x0 = E[a] + J(E[b])
r = RMSNorm(x0)
z[c] = α[c] r[c] + β[c] r[(c + 1) mod W]
x1 = x0 + z²
logits = RMSNorm(x1) Eᵀ
```

Square follows the weighted sum. This matters for both forward values and
edge gradients. The sole mesh has independent signed `α` and `β`, and no
bias. RMSNorm uses all channels, epsilon 1e-5, and no learned affine
parameters. Unique parameters total `97W + 2W = 99W`.

The fixed role rotation draws on [RoFormer's pairwise rotations](https://arxiv.org/html/2104.09864v5).
Its angle depends only on operand role; the model has no attention or
residue-dependent rotation frequencies. The `a` lookup, rotated `b` lookup,
and readout gradients accumulate into the same `E`, with one AdamW state per
unique parameter. SplitMix64 initializes `E` uniformly in
`[-1/√97, +1/√97]`; identity mesh edges start at `α=1, β=0`.

The **All** diagram in Network draws every learned connection, with white nodes
and signed colored arrows inspired by Wolfram's
[`sw08202024topologyimg1`](https://content.wolfram.com/sites/43/2024/08/sw08202024topologyimg1.png).
`E` appears in the `a`, rotated `b` and readout roles, so there are **37,504**
displayed connections at W128 and **300,032** at W1024, while unique parameter
counts remain 12,672 / 101,376. Pink means positive, blue negative, and line
width and opacity encode magnitude. Dense edges use low opacity. Every
connection is included; Highlight is off by default and can emphasize the
selected channel without removing other edges. Network opens a larger diagram.
Wheel/buttons zoom horizontally
so every layer remains visible; dragging pans,
Fit restores the overview, and left/right arrow keys select circular channels.

The network has independent **Weights** and **Change ×4** modes. Change shows
`weight - initialization` in all three E roles and both Mesh edges; positive
changes are pink, negative blue, and zero change is transparent. Its visual
scale is four times more sensitive. Both modes use fixed initial scales and a
smooth magnitude mapping, so an increasing current maximum cannot normalize
away growth. All real snapshots use the selected 1/5/10 Hz capture cadence.
Mode changes, Highlight and zoom preserve the model and reuse uploaded weights.
WebGL2 accumulates transparent lines in an RGBA16F target when supported,
then composites once. This preserves weak changes that were rounded away when
each line blended directly into RGBA8. Change uses Canvas when float targets
are unavailable. Canvas fallback bins magnitudes and uses
a small visibility floor for nonzero edges.

**Channel** expands the selected branch into a **2 → 1** diagram: three
nodes, two learned edges, Square at the output, and the residual skip. Node
centers show actual probe activations. RMSNorm depends on the complete vector;
the All diagram labels these fixed operations rather than drawing their global
dependencies as learned edges. The atlas selector shows shared `E`, derived
`J(E)` and tied `Eᵀ`, or the mesh's **one-row** self/neighbor weights.

Click nodes, heatmaps, or the contribution plot to select a channel. The
numeric selector reaches every channel, including wraparound channels 127
and 1023. If several channels share one plot pixel, the bar shows their
maximum contribution and selects its actual channel. Raw weights, changes
from initialization, and centered-readout views remain available.

## Prediction dynamics

The default Dynamics view shows the **logit margin** for 128 fixed equations
from the selected seed's held-out split:

```text
margin = logit[correct answer] − max(logit[incorrect answers])
```

Positive margins guarantee that the correct answer wins; negative margins
mean that an incorrect answer wins; zero is a tie. Pink turns green as each
equation starts to generalize. The fixed color scale is −4…+4, with stronger
values clipped. The histogram includes those tails in its first and last bins.
Each heatmap column is a real snapshot, with its actual epoch labeled below.
Columns are equally spaced by snapshot sequence, rather than by elapsed
epochs; skipped epochs are not fabricated or interpolated. Hover reports the
historical value; click or up/down keys select a row and show its latest
prediction, margin, and softmax probability of the correct answer.

The sample is chosen once from the shuffled held-out split, then sorted by
answer for display. Its accuracy is labeled **Sample**; the learning curves
and Training desk still use all 4,656 examples in each split every epoch.
Diagnostic snapshots and complete-split observations have separate timestamps.
Switching model tabs retains independent histories; Reset clears the selected
model's history. Diagnostics are recorded only while their view is visible.
Their history is bounded to 384 snapshots, thinning older captures as needed
while retaining the first and latest.

Diagnostic inference runs in a separate CPU worker using the existing weight
captures. It calculates only 128 equations, keeps E dot products contiguous,
and submits no extra training or evaluation passes. Busy workers finish the
current snapshot before reading the latest one. The heatmap bitmap is cached
between snapshots; metric updates and resizing reuse it. Diagnostic inference
uses Float64 arithmetic on captured FP32 weights, like the existing probe;
an almost tied prediction can differ from the training engine's FP32 result.

The [live browser check](../web/own-channel/tests/results/rotary-prediction-dynamics-verification.json)
trained both widths with Dynamics visible at 10 weight captures per second:

| Width | Final complete-split accuracy | Epoch | Sample correct | Median diagnostic compute time |
| --- | ---: | ---: | ---: | ---: |
| 128 | 99.01% | 113 | 128 / 128 | 1.7 ms |
| 1024 | 99.03% | 181 | 127 / 128 | 10.9 ms |

On W1024, the sample's median margin moved from −5.04 initially to −3.73 at
captured epoch 40, −1.77 at 100, and +9.13 at 181. The
[final history](../web/own-channel/tests/results/rotary-prediction-dynamics-final.png) contains
93 real snapshots. Worker timings exclude transfer and drawing, on the local
Ryzen 5 5600H; the full W1024 UI run took 13.07 seconds including polling and
screenshots. Six diagnostic checks passed against independent model inference,
and all four views passed 34 viewport checks, including 320×568. GPU and WASM
training, switching, reset, and the existing complete-network view were checked.

Raw edge colors obscure this change: 97 of every 99 parameters belong to E,
and its three displayed roles overlap heavily. The existing all-weight graph
and atlases remain available to inspect the architecture. Seed-11 final-weight
[analysis](../web/own-channel/tests/results/rotary-m1-square-visualization-analysis.json) also found
that W1024's five strongest discrete-log Fourier frequencies carry only
13.4% of centered nonzero-residue embedding energy (11.1% initially).
Embedding effective rank stays around 91.4, despite ≥99% test accuracy;
therefore a handful of embedding Fourier circles is not assumed to explain
this model. W128 concentrates more strongly, from 11.8% to 31.4% in five
frequencies. These are descriptive endpoint measurements, not causal circuit
ablations. For a different model, see Nanda et al.'s
[progress measures for modular-addition Transformers](https://arxiv.org/abs/2301.05217).

The minimal W1024 model also has an actual 109-epoch 10%→90% rise, rather than
a one-frame transition. Dynamics visualizes its gradual generalization; it
does not change the architecture or relax the grokking criterion.

## Protocol and measurements

Both tabs default to seed 11: 9,312 division-modulo-97 pairs, a fixed 50%
split, 4,656 pairs per split, nine batches of 512 and a 48-row tail per epoch.
AdamW uses learning rate 0.001, decay 0.1, betas (0.9, 0.98), epsilon 1e-8,
and ten warmup updates. Both complete splits are evaluated **every epoch**.
The step button advances one epoch. The cap is 10,000 epochs, with optional
stopping at the first held-out accuracy ≥99%. Curves contain actual
observations; each engine trains from random initialization.

Fresh seed-11 measurements of the minimal model:

| Width | Rust test ≥99% | WASM SIMD test ≥99% | WebGPU test ≥99% | WASM training + evaluation | WebGPU training + evaluation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 113 epochs | 113 epochs | 113 epochs | 6.17 s | 1.32 s |
| 1024 | 181 epochs | 181 epochs | 181 epochs | 72.18 s | 12.50 s |

W1024 observes train ≥99% / held-out ≤10% at epochs 26–38 (**12 epochs**) in
all three engines, then first held-out ≥90% at 147: a **109-epoch** rise.
There is no perfect-train plateau under that test threshold. W128 has no
memorization plateau and reaches ≥90% at epoch 95, a 72-epoch rise.

The grokking label requires at least 25 contiguous epochs with train ≥99%
and held-out ≤10%, followed by a ≤250-epoch rise to ≥90% and later ≥99%.
Train dips or missing observations break the plateau. Beige shading marks
train ≥99%; violet marks train =100%. The criterion is preserved even though
the simpler model does not meet it on the five tested seeds.

WASM timings use one thread on an AMD Ryzen 5 5600H in Node v24.21.0. WebGPU
uses a hardware adapter reporting `fallback: false`; its description is
hidden. Times exclude engine setup and normal live weight readbacks. Different
FP32 reductions and WASM softmax approximations can move crossing epochs.

Current histories and source hashes are in
[`neighbor-rotary-m1-square-wasm-simd-seed11`](../web/own-channel/tests/results/neighbor-rotary-m1-square-wasm-simd-seed11/run.json),
[`neighbor-rotary-wide-m1-square-wasm-simd-seed11`](../web/own-channel/tests/results/neighbor-rotary-wide-m1-square-wasm-simd-seed11/run.json),
[`neighbor-rotary-m1-square-webgpu-seed11`](../web/own-channel/tests/results/neighbor-rotary-m1-square-webgpu-seed11/run.json), and
[`neighbor-rotary-wide-m1-square-webgpu-seed11`](../web/own-channel/tests/results/neighbor-rotary-wide-m1-square-webgpu-seed11/run.json).
Older histories retain their original filenames and source hashes.

## Efficient execution

- Rotate90 uses paired shuffles and signs, without trigonometry or a learned
  input projection. The readout shares the embedding table.
- WebGPU keeps parameters, gradients, moments, and activations on the device.
  Deterministic reductions avoid float atomics; readout gradients use tiled
  matrix products; all three `E` roles merge before AdamW.
- The minibatch cache uses **five planes**, reusing the mesh preactivation
  for its derivative and the final residual state for the input derivative
  after their last reads. This is **1.25 MiB** at W128 and **10 MiB** at W1024,
  down from 4.25 and 34 MiB. Complete-split evaluation uses workgroup-local
  states. Wide kernels use 128 threads with eight channels each and less than
  16 KiB of workgroup storage.
- Up to 32 updates share a command buffer; submissions adapt toward 50 ms.
  CPU workers yield through MessageChannel so Pause remains responsive.
- Weight snapshots are 50,688 bytes at W128 and 405,504 at W1024. Default 5 Hz
  readbacks transfer about 0.25 or 2.03 MB/s. Hidden views do not draw;
  heatmaps use pixel buffers and probes use the latest actual weights.
- Complete-network drawing uses one WebGL2 instanced batch and a composite
  pass. An f32 texture stores current unique parameters; an initial texture is
  uploaded once per model/seed for Change. Vertex IDs generate all three E
  roles and both mesh edges. Zoom/pan reuse the textures without another upload.
  The diagram draws only on changes. A batched Canvas 2D fallback includes
  every connection when WebGL2 is unavailable.

The earlier [drawing measurement](../web/own-channel/tests/results/rotary-m1-square-all-weights-performance.json)
rendered all 300,032 W1024 connections at 1250×520: a 0.4 ms median on the
integrated AMD Radeon OpenGL adapter, and 25.7 ms for the Canvas fallback.
This standalone arrow measurement includes texture upload and `gl.finish`;
it excludes probe inference, nodes, training and other panels. The GPU
median excludes two warmup draws. Dense E connections overlap even in the
full [overview](../web/own-channel/tests/results/rotary-m1-square-all-weights-overview.png);
[horizontal zoom](../web/own-channel/tests/results/rotary-m1-square-all-weights-zoom.png) and
channel emphasis make individual Mesh arrows visible.

The visuals follow Stephen Wolfram's [2024 minimal-models article](https://writings.stephenwolfram.com/2024/08/whats-really-going-on-in-machine-learning-some-minimal-models/).
Readout centering removes the class-independent component ignored by softmax.

## Handwritten WebAssembly and checks

The current kernels are handwritten WAT in `wasm/neighbor-rotary.wat` and
`wasm/neighbor-rotary-wide.wat`. Shared scalar/SIMD matrix and rotation routines
are inserted at build time. Earlier trainers remain as regression baselines.
The modules have no imports, allocator, threads, shared memory, or memory growth.

| Width | SIMD binary | Scalar binary | Fixed linear memory |
| --- | ---: | ---: | ---: |
| 128 | 3,634 bytes | 3,174 bytes | 448 KiB |
| 1024 | 3,652 bytes | 3,192 bytes | 2.50 MiB |

WABT is the only build dependency and is not shipped to the browser. From
this directory:

```sh
nvm use
npm ci --ignore-scripts
npm run build:wasm
npm test
```

All **74 Node tests** passed. They cover Rust checkpoint parity, dataset and
sampling, finite differences with signed Square outputs, shared-E gradients
when operands coincide, RMSNorm/skip derivatives, circular edges, every
parameter gradient for batches 1/37/48/512, and scalar/SIMD AdamW. WebGPU
passed **18 f64 gradient/optimizer checks per width**. UI checks cover live
weights, loss/accuracy, pause/reset, independent tabs, wraparound channels,
the single mesh's atlases, model information, and viewport fit from 320×568
to 1920×1080.

The [numerical verification](../web/own-channel/tests/results/rotary-m1-square-webgpu-verification.json)
records unchanged training sources. The initial
[browser verification](../web/own-channel/tests/results/rotary-m1-square-browser-verification.json)
archives the first complete-network view. Current
[dynamics verification](../web/own-channel/tests/results/rotary-all-weights-dynamics-verification.json)
records Weights/Change modes, fixed scales, optional Highlight, all 26 viewport
checks, and a complete W1024 WebGPU run through 99.03% held-out accuracy at
epoch 181. Its snapshots at initialization, epochs 41/101 and the target show
all connections with no channel emphasis. Arrow rendering with RGBA16F
accumulation measured a 0.3 ms median; Canvas took 27.4 ms, under the same
standalone procedure described above. A second browser script
measures drawing and checks actual mouse panning, Fit, texture reuse and the
Canvas fallback:

```sh
GROKKING_CDP_PORT=9224 node web/own-channel/tests/measure-topology.mjs
```

From the repository root, with an isolated Chrome profile on CDP port 9224:

```sh
GROKKING_PREVIEW_ORIGIN=http://127.0.0.1:8080 \
  node web/own-channel/tests/browser.mjs
```

Downloaded weights are raw little-endian f32: shared `E`, then one mesh's
interleaved self/neighbor weights. Metrics export as CSV. Weight exports
omit AdamW state and cannot resume an optimizer run.
