# Rotate90 Dense Grokking Lab

Two local tabs train one Dense + ReLU architecture at widths 128 and 1024.
Training runs entirely in the browser on handwritten WASM SIMD128 or scalar
WASM. Training and weight rendering use CPU workers. Node serves static files.

| Tab | Channels | Unique parameters | Drawn learned connections |
| --- | ---: | ---: | ---: |
| Rotate90 · 128 | 128 | 28,800 | 53,632 |
| Rotate90 · 1024 | 1024 | 1,147,904 | 1,346,560 |

## Build and run

From the repository root:

```sh
cd web/own-channel
nvm use
npm ci
npm run build
npm start
```

Open **http://127.0.0.1:8080**. `.nvmrc` selects Node v24.21.0. The preview
binds to localhost. Training is independent of the static preview server.

The frontend, preview server, build tools and tests are TypeScript. The build
runs the pinned TypeScript compiler with `strict` and `noEmitOnError` for both
browser and Node code, then assembles WASM and generates model SVGs. Sources
live in `src/`; browser JavaScript, declarations and source maps are emitted
into `dist/`. Node utilities and browser test modules are emitted into ignored
`build/`. Source maps embed the original TypeScript. There is no runtime
TypeScript compiler, ML framework or bundler.

```sh
npm run typecheck
npm test
```

`npm test` builds first and tests the emitted JavaScript and assembled binaries.
Rebuild after editing sources; `npm run typecheck` checks an existing build
without emitting code. The fixed dependency versions are in `package-lock.json`.

Training is the default workspace. Weights and Network provide the other two
views. Both model tabs preserve their independent weights, AdamW state, sampler
and curves. Switching tabs pauses training; changing seed or backend resets
that tab. Retired browser architectures and the Dynamics view are removed from
active code and build inputs. Native experiment archives remain available in
[the historical studies](../../docs/grokking-browser-models.md).

## Architecture and initialization

```text
E ∈ R^(97 × W), A ∈ R^(W × W), W ∈ {128, 1024}
J(u, v) = (-v, u) on adjacent coordinate pairs
x0 = E[a] + J(E[b])
r = RMSNorm(x0)
z = r A
x1 = x0 + ReLU(z)
h = RMSNorm(x1)
logits = h Eᵀ
prediction = smallest class attaining the largest logit
```

RMSNorm is pre-norm inside the branch. The skip carries the original x0; a
separate final RMSNorm precedes readout. Both use epsilon 1e-5, without learned
gain. There are no biases. One E supplies the a lookup, rotated b lookup and
readout; all three gradients share one parameter and one AdamW state.

E starts **Normal(0, variance 1/W)**; A starts Kaiming normal with variance
2/W. Class embedding norms have expectation near one and are free to change
in training. SplitMix64 and Box–Muller match the native initial checkpoints
byte-for-byte at seed 11:

- W128: `b90448461caff3b6e04bd9ead0a3f7e7f055e8c1cd71ce19f1069186c74ae04b`
- W1024: `ea659d5b1596d3a019c3d2af55cb4c030244481b73011b0def3572ddc04d1e91`

The initializer consumes the same number of random draws as the old uniform
E, preserving every A weight. See the
[paired Rust initialization study](../../docs/rotary-dense-embedding-results.md).

## Task and training

The task is division modulo 97, with a ∈ {0,…,96}, b ∈ {1,…,96} and
q = a·b⁻¹ mod 97. There are 9,312 equations: 4,656 train and 4,656 test.
Seed 11 is the default. Every epoch visits all training rows in nine 512-row
batches and a 48-row tail. AdamW uses lr 0.001, weight decay 0.1, betas
0.9/0.98, epsilon 1e-8 and ten warmup updates. Both complete splits are
evaluated every epoch. Step advances one epoch; stopping at 99% test accuracy
is optional. The limit is 10,000 epochs.

The grokking criterion stays fixed: at least 25 continuous elapsed epochs
with train ≥99% and test ≤10%, then test ≤10% → ≥90% within 250 epochs and
a later test ≥99% result. The first 90% crossing is frozen; later dips cannot
improve that measured rise. Train dips and missing observations break a plateau.
Beige marks train ≥99%; violet marks perfect train, both with test ≤10%.

## Visualization and languages

Training shows realtime complete-split loss and accuracy. Weights shows E,
the complete W × W matrix A and the tied readout. Raw values, changes from
initialization and centered readout are available. Its pink/blue palette uses
1,024 magnitude levels per sign with the same fixed logarithmic scale as Network,
including Change ×4. Exactly zero values have a neutral background. Centering
the readout keeps the initial E scale. The footer reports the current maximum
magnitude, rather than the color-scale limit. The palette is built once, and
heatmap rasters are cached until weights or display mode change. Contiguous
single-pass rasterization reuses its pixel buffers between snapshots.

Network renders every A connection and every E connection in its three shared
roles with software Canvas 2D in an OffscreenCanvas worker. Magnitude uses `log1p(abs(weight) / initialRMS)`,
mapped to `strength / (1 + strength)`. E and A have separate initial RMS references
that stay fixed throughout training. Change ×4 uses one quarter of those references;
exactly unchanged weights are transparent. Nonzero changes use a single magnitude
factor for opacity, so small updates stay visible.
The legend and both views share the same color stops. Network uses
1,024 magnitude bands per sign. Its software WASM rasterizer accumulates every
line's scanline spans in an f32 difference buffer, then combines overlaps into
pixels. Color is a mass-weighted average; opacity is mass / (1 + mass).
This avoids Canvas path intersection costs and RGBA8 rounding of weak edges.
Weight classifications and the background raster are cached; highlighting
reuses the background. Browsers without OffscreenCanvas use the same renderer on
the main thread. Highlight retains all edges; zoom, drag, Fit and arrow keys
inspect the network. Channel shows all W inputs to one ReLU output and its skip.
All uses neutral nodes with stable styling. Channel nodes show probe activations.
Only visible panels whose inputs changed draw. Rendering is scheduled on demand
and sleeps while idle or hidden. Metric messages retain every epoch but update
the UI at most 10 Hz; weight snapshots use the selected 1/5/10 Hz cadence.
Hover tooltips do not repaint the atlas. Probe predictions and readout scales are
cached independently of pan, selection and metrics. Unchanged text stays in the
DOM, and the worker never resends a weight snapshot at the same training step.

Weight capture defaults to 5 Hz, with 1/10 Hz choices. Snapshots are 115,200
bytes for W128 and 4,591,616 for W1024. Adaptive training submissions target
50 ms; CPU workers yield between submissions for responsive pause. Metrics
export as CSV; weight exports contain little-endian f32 E followed by A and
omit optimizer state.

The startup dialog explains the task, inverse-based solution, eight checked
examples and the actual model. It includes a local rendered Mermaid SVG and
native model SVGs, tensor types and quantified equations. The ∑ button reopens
it. English, Spanish, German, Japanese, French and Russian are available.
Automatic selection follows browser preference order and regional tags;
manual choice persists locally. Language changes preserve training.

Translation sources are `scripts/translations.ts`. Model SVGs are generated
by `scripts/build-model-diagrams.ts`. Mermaid sources and hashes are in
`diagrams/`; regenerate them with `npm run build:diagrams` using an isolated
Chrome profile at `GROKKING_CDP_PORT`. Mermaid is not loaded at runtime. The
128-channel WAT is the source template in `wasm/`; the build specializes it
for 1024 channels and emits SIMD/scalar binaries with private or shared memory.

The engine selector exposes 1, 2 or 4 CPU threads for both WASM variants.
Automatic selection uses four threads with at least six logical CPUs, two
with four or five, and one on smaller devices.
Parallel training partitions each batch, sums private gradients in fixed slot
order and performs one AdamW update on shared weights. Evaluation partitions
the complete split and combines f64 loss sums and integer correct counts.
Floating-point gradient regrouping can shift convergence epochs compared with
a single thread. Shared memory requires cross-origin isolation; the local
server supplies COOP/COEP headers. Without that capability the app uses one
thread. SIMD support is detected automatically, with scalar fallback.

## Validation and observed convergence

The TypeScript build and 41 Node tests pass: deterministic data and sampling,
strict grokking classification, native checkpoint hashes, scalar/SIMD logits,
all parameter gradients, shared E accumulation, finite differences and AdamW.
Two- and four-thread variants also pass gradient, AdamW and complete-split
evaluation checks for both widths and batch sizes including 1, 48 and 512.
Scalar/SIMD raster tests verify actual pixels for all three E roles, Rotate90
signs, Dense connections, zero changes and offscreen clipping.
The real UI trains both widths with GPU APIs forbidden,
pauses/resets, restores independent states, updates weights, and fits all three
views. Localization checks pass 216 desktop/mobile dialog and workspace layouts.

Historical convergence measurements (GPU engines have since been removed):

| Width / engine, seed 11 | First test ≥90% | First test ≥99% | Memorization plateau | Rise to ≥90% | Grokking |
| --- | ---: | ---: | --- | ---: | --- |
| 128 / Rust CUDA | 510 | 605 | None | 203 epochs | No |
| 128 / Node WASM SIMD | 520 | 617 | None | 214 epochs | No |
| 128 / Chrome WebGPU | 509 | 606 | None | 207 epochs | No |
| 1024 / Rust CUDA | 226 | 270 | 18–110: 92 epochs | 116 epochs | Yes |
| 1024 / Chrome WebGPU | 292 | 406 | 18–111: 93 epochs | 181 epochs | Yes |

These are one-seed measurements. Floating-point reduction and softmax differences
change crossing epochs despite identical initialization. W1024 WASM passes
numerical and real UI checks; a complete W1024 WASM convergence run is not measured.

Observed training plus evaluation: W128 WASM 65.45 s, W128 WebGPU 8.19 s,
W1024 WebGPU 143.10 s. WASM used Node v24.21.0 on a Ryzen 5 5600H; WebGPU used
Chrome 155 on NVIDIA Ampere with a hardware adapter. These runtimes exclude
setup and live dashboard rendering; separate CPU/UI checks ran during some
measurements, so the numbers are not isolated comparative benchmarks.
Standalone all-edge drawing at 1250×520 measured 3.3 ms GPU median on AMD
integrated graphics and 115.3 ms Canvas fallback, excluding node overlay and
training.

Current artifacts and histories:

- [W128 WASM SIMD](tests/results/dense-rotary-normal-width-wasm-simd-seed11/run.json)
- [W128 WebGPU](tests/results/dense-rotary-normal-width-webgpu-seed11/run.json)
- [W1024 WebGPU](tests/results/dense-rotary-wide-normal-width-webgpu-seed11/run.json)
- [Build, numerical and UI checks](tests/results/normal-width-typescript/)

GPU studies preserve the exact measured source files under their `source/`
directories. Previous uniform-E and Neighbor measurements remain historical;
[archived Neighbor documentation](../../docs/browser-neighbor-history.md)
describes the retired frontend.

To rerun browser checks after building, use an isolated Chrome profile on the
selected CDP port. Hardware acceleration is not required:

```sh
GROKKING_CDP_PORT=9224 npm run test:browser
GROKKING_CDP_PORT=9224 npm run test:rendering
GROKKING_CDP_PORT=9224 npm run test:i18n
GROKKING_CDP_PORT=9224 node build/tests/measure-topology.js
node build/scripts/measure-cpu.js /tmp/grokking-cpu-benchmark
node build/scripts/measure-wasm.js dense-rotary simd tests/results/new-wasm-run 11
```
