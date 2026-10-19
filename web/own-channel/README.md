# Rotate90 Dense Grokking Lab

Five preset tabs train one Dense + ReLU architecture at widths 16, 32, 64, 128
and 256. Every width has a fast and a slow optimizer preset; W128 and W256 add a
plateau preset.
Training runs entirely in the browser on handwritten WASM SIMD128 or scalar
WASM. Training and weight rendering use CPU workers. All five widths have
prebuilt WASM modules; both local preview and GitHub Pages serve static files.

| Tab | Channels | Unique parameters | Drawn learned connections |
| --- | ---: | ---: | ---: |
| W16 | 16 | 1,808 | 4,912 |
| W32 | 32 | 4,128 | 10,336 |
| W64 | 64 | 10,304 | 22,720 |
| W128 | 128 | 28,800 | 53,632 |
| W256 | 256 | 90,368 | 140,032 |

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

The **Settings** workspace adds a custom experiment at width 16, 32, 64, 128 or
256, with a positive learning rate and nonnegative weight decay.
**Apply and open training** creates a fresh experiment using the
current seed, engine, epoch limit and snapshot cadence. Its **Custom** model tab
preserves weights, optimizer and curves when switching to presets and back.
Applying again replaces only the custom experiment. Reset and PyTorch exports
retain its width, lr and wd.

Every control group, metric and chart has a **?** button. The 31 help topics
explain the current display in short paragraphs, with a link to the relevant
introduction page where more detail is useful. The same dialog supports all
six languages, touch, keyboard activation and Escape, and restores focus on
close. Help does not change experiment settings or start training.
On phones shorter than 701 CSS pixels, the workspace scrolls internally to
keep plots readable; the header and workspace tabs remain visible.

The width selector uses exactly the five prebuilt architectures. Every width
supports SIMD and scalar WASM, private memory and shared CPU workers. Learning
rate and weight decay are supplied to AdamW at runtime.

## GitHub Pages

The [published lab](https://fxeqxmulfx.github.io/grokking_abmod/) is deployed by
[the Pages workflow](../../.github/workflows/pages.yml) on pushes to `master`
that change the site, its license or the workflow. It can also be run manually
from GitHub Actions. The workflow uses Node from `.nvmrc`, installs the lockfile,
builds and tests, packages `build/pages/`, then deploys that artifact. Official
Actions are pinned to commit hashes.

To preview the publication locally:

```sh
npm run build:pages
npm run preview:pages
```

Open `http://127.0.0.1:8081/grokking_abmod/`. This preview serves the artifact
under the project prefix without special HTTP headers. The scoped isolation
worker supplies COOP/COEP headers so Pages can use shared-memory CPU workers.
The first visit may reload once before the app starts. Browsers that cannot
use service workers retain the single-thread fallback. The worker fetches from
the host and does not maintain an asset cache.

The source and packaged site include the [MIT license](../../LICENSE).

Training is the default workspace. Weights and Network provide the other two
views. Every width and optimizer preset preserves its independent weights,
AdamW state, sampler and curves. Switching experiments or presets pauses training;
changing seed or backend resets that experiment. Retired browser architectures
and the Dynamics view are removed from active code and build inputs. Native
experiment archives remain available in
[the historical studies](../../docs/grokking-browser-models.md).

Only the active experiment retains a training worker and its CPU thread pool.
On touch devices, leaving Training while paused also releases that pool;
returning restores it. Inactive experiments keep compact checkpoints containing
weights, both AdamW moments, bias corrections, shuffled sampler state and the
grokking criterion.
Returning restores those states without adding a duplicate observation.

Hidden canvases release their backing stores and keep 1×1 placeholders. The
network renderer starts only when visible and is terminated on leaving the
view, releasing its WASM memory and intermediate image. The visible canvas
stays on the main thread and receives transferable RGBA frames from the worker;
it never relies on implicit OffscreenCanvas presentation. The worker waits for
presentation acknowledgement before sending another image, bounding queued
frame buffers to one. Resize observation uses stable containers. Visible display
surfaces render at up to 2× CSS resolution, with a 524,288-pixel budget per
mobile surface and 1,048,576 pixels on desktop. The network raster is limited
to 524,288 pixels and reuses strips of at most 131,072 pixels for f32
accumulation, keeping its WASM memory below 4 MiB even at the higher resolution.
Heatmaps use nearest-neighbor scaling and paint immediately while a trainer
checkpoint is pending; only the dense renderer waits for trainer shutdown.
Restored canvas contexts invalidate their display and bitmap caches. The
renderer keeps all connections and restores the selected channel, zoom and pan
when reopened.

## Architecture and initialization

```text
E ∈ R^(97 × W), A ∈ R^(W × W), W ∈ {16, 32, 64, 128, 256}
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

- W16: `2b063d5498dd54fec02097d17561b06fe977a41c6251730e2a424227e382c3c4`
- W32: `e81b4ca9d1d0625cbdcde867a6e22bb3ec84140651dcbe1b355ee1c32c36fe39`
- W64: `4433d6019f75d87ae00108d5befb6474a9e0bd4d2df65ce0358677c1b365ff80`
- W128: `b90448461caff3b6e04bd9ead0a3f7e7f055e8c1cd71ce19f1069186c74ae04b`
- W256: `a2b331ff08b75dbb3c397e85b94769f3f4ab973baa9877c24217046652263468`

The initializer consumes the same number of random draws as the old uniform
E, preserving every A weight. See the
[paired Rust initialization study](../../docs/rotary-dense-embedding-results.md).

## Task and training

The task is division modulo 97, with a ∈ {0,…,96}, b ∈ {1,…,96} and
q = a·b⁻¹ mod 97. There are 9,312 equations: 4,656 train and 4,656 test.
Seed 11 is the default. Every epoch visits all training rows in nine 512-row
batches and a 48-row tail. AdamW takes its learning rate and weight decay from
the [selected preset](#optimizer-presets), with betas 0.9/0.98, epsilon 1e-8
and ten warmup updates. Both complete splits are evaluated every epoch. Step
advances one epoch. The first test accuracy ≥99% crossing automatically pauses
training. Continue resumes the same weights, AdamW state and sampler; the target
does not pause that experiment again, even after an accuracy dip. Reset re-arms
it. The limit is 10,000 epochs.

The original paper is [Grokking: Generalization Beyond Overfitting on Small
Algorithmic Datasets](https://arxiv.org/abs/2201.02177). The browser uses the
Dense architecture above rather than the paper's Transformer.

[Small-width Rust results](../../docs/rotary-dense-small-width-results.md)
with the earlier lr 0.001 and weight decay 0.1, seed 11: W32 reaches 99% test
accuracy at epoch 3690. W16 ends at 32.09% train and 18.11% test after 10,000
epochs. W64, tested in Rust, reaches 99% test at 560, compared with 605 for
W128. Width alone changes in these experiments; other seeds or floating-point
backends can produce different results.

## Optimizer presets

**Fast**, at every width, is the winner of the native Rust/CUDA optimizer grid:
the earliest 99% test accuracy over seeds 11, 17 and 23. **Slow** comes from a
late-generalization search. Of the configurations that reach 99% on all three
seeds within 10,000 epochs, it is the one whose earliest crossing comes latest.
**Plateau** is the one among them with the longest memorization plateau, by the
grokking criterion below, on its shortest seed. At W256 that is the slow
configuration, so Plateau takes the longest plateau followed by grokking on
every seed instead. No configuration up to W64 holds a plateau on every seed.
[CUDA Dense products](../../docs/dense-cuda-kernels.md#optimizer-search)
describes the searches and their archives. The header shows the selected
preset's learning rate and weight decay next to the task.

| Tab | Preset | lr | Weight decay | Rust/CUDA first test ≥99%, seeds 11 / 17 / 23 | Rust/CUDA plateau, seeds 11 / 17 / 23 | Node WASM, seed 11: first test ≥99% (plateau) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| W16 | Fast | 0.1 | 0 | 1,033 / 407 / 163 | None | 996 |
| W16 | Slow | 0.2 | 0.001 | 5,182 / 1,333 / 7,117 | None | 5,118 |
| W32 | Fast | 0.1 | 0.0001 | 39 / 43 / 27 | None | 39 |
| W32 | Slow | 0.0003 | 0.1 | 5,199 / 3,885 / 5,462 | None | 5,382 |
| W64 | Fast | 0.1 | 0.1 | 21 / 33 / 29 | None | 21 |
| W64 | Slow | 0.0001 | 0.0001 | 6,513 / 6,583 / 8,501 | None | 6,426 |
| W128 | Fast | 0.01 | 1 | 30 / 32 / 30 | None | 30 |
| W128 | Plateau | 0.0003 | 0 | 3,375 / 3,770 / 2,880 | 166 / 150 / 210 | 3,729 (173) |
| W128 | Slow | 0.00005 | 0.3 | 8,151 / 7,931 / 8,899 | None | 8,189 |
| W256 | Fast | 0.01 | 1 | 32 / 33 / 27 | None | 32 |
| W256 | Plateau | 0.001 | 0.3 | 368 / 346 / 336 | 76 / 69 / 58 | 366 (82) |
| W256 | Slow | 0.00005 | 0.3 | 7,632 / 6,777 / 6,593 | 1,644 / 1,403 / 1,362 | 7,737 (1,664) |

The W256 Plateau rises from its plateau to 90% test accuracy within 115–129
epochs on every seed, which the criterion counts as delayed grokking. The W128
Plateau and W256 Slow rise over 892–1,362 and 2,126–2,721 epochs.

The Node column ran the browser's four-thread WASM SIMD engine to the first
99% crossing. Thread partitioning regroups floating-point sums and can shift
that epoch; with one thread, W16 fast crosses at 1,033, as on CUDA. The high
learning rates make W16 sensitive to such differences. W16 slow crosses at
5,070–6,930 on seed 11 with one, two or four threads, and at 5,391 and 6,643 on
seed 23 with one and four. Seed 17, which crosses at 1,333 on CUDA, stays below
99% for all 10,000 epochs with one, two and four threads. W128 Plateau crosses
354 epochs after CUDA on seed 11.

## Offline SwiGLU and ScheduleFree comparison

The 2026-10-20 CUDA comparison replaces the residual branch with
`(SiLU(r G) * (r U)) D`, using hidden width 43 at W128 and 85 at W256.
Parameter counts are 28,928 and 90,112, compared with ReLU's 28,800 and
90,368. E, its initialization, the division task, split and minibatch orders
are shared. G starts with variance 2/W, U with 1/W and D with 1/hidden.

The actual model and `ScheduleFreeAdamW` come from the local `transformer`
checkout at `4f764a4d08420f73af1fe8672deda49461e7c149`. All 24 confirmation
runs complete 1,000 epochs with seeds 11, 17 and 23, weight decay 1, betas
0.9/0.98, epsilon 1e-8 and ten inclusive warmup updates (one epoch).
PyTorch 2.14.1+cu126 uses deterministic FP32 operations with TF32 disabled;
AdamW uses `foreach=False`. Full splits are evaluated every epoch.
The seed-11 pilot also participates in confirmation; seeds 17 and 23 are
the additional seeds after selecting learning rates.

| SwiGLU optimizer | lr | First test >=99%, W128, seeds 11 / 17 / 23 | First test >=99%, W256 | Later observations below 99% |
| --- | ---: | --- | --- | --- |
| AdamW | 0.01 | 14 / 15 / 16 | 15 / 17 / 14 | Yes; worst test accuracy 69.65% |
| AdamW | 0.003 | 30 / 35 / 30 | 36 / 35 / 35 | None |
| ScheduleFreeAdamW | 0.01 | 27 / 32 / 26 | 32 / 29 / 29 | None for averaged weights |
| ScheduleFreeAdamW | 0.03 | 13 / 19 / 14 | 14 / 15 / 12 | None for averaged weights |

ScheduleFree results evaluate its averaged weights with `optimizer.eval()`;
`optimizer.train()` restores the query before the next training forward.
The query still has losses of fit, reaching 19.24% held-out accuracy at
lr 0.03. Averaging supplies the observed evaluation stability through this
finite budget. Earlier ReLU runs at the same ScheduleFree lr 0.03 first
reached 99% at 27 / 28 / 26 epochs (W128) and 28 / 27 / 27 (W256).
No confirmation run has the browser criterion's 25-epoch memorization
plateau. Every final evaluated model reaches 100% on both splits.

[Comparison JSON](../../docs/rotate90-swiglu-results/comparison.json),
[per-seed CSV](../../docs/rotate90-swiglu-results/summary.csv),
[original environment and source hashes](../../docs/rotate90-swiglu-results/metadata.json)
and the [original local artifact manifest](../../docs/rotate90-swiglu-results/artifact-manifest.json)
retain compact evidence. The original complete bundle is local at
`/tmp/grokking-swiglu-20261020/`. Its 24 saved checkpoints reproduce their
metrics exactly, including both ScheduleFree weight modes; replay of all
36 trajectories agrees with the browser classifier. Full curves and
checkpoints stay local.

To regenerate from the repository root, use the pinned transformer checkout.
`ROTATE90_TRANSFORMER_ROOT` defaults to `~/lean_projects/transformer`:

```sh
uv run --locked scripts/compare-rotate90.py
uv run --locked scripts/compare-rotate90.py --verify
node scripts/verify-rotate90-criterion.mjs target/rotate90-swiglu
```

Outputs default to ignored `target/rotate90-swiglu/`. A short training check is:

```sh
uv run --locked scripts/compare-rotate90.py \
  --plan docs/rotate90-swiglu-results/smoke.json \
  --output target/rotate90-swiglu-smoke
```

The committed harness removes temporary helper imports and keeps the original
training operations. Its checkpoint verification reproduces the retained run
metrics exactly. Epoch counts compare this PyTorch backend; they are not
wall-clock speed measurements or byte-identical browser trajectories.

## PyTorch source export

The **PyTorch .py** button exports a standalone CPU training script with the
selected width, optimizer preset, seed and epoch limit. It includes the model,
exact initialization, dataset split, shuffled epoch sampler, AdamW settings and
CSV metrics. Run the
download with `uv run rotate90-w128-seed11.py` (Plateau and Slow presets add
`-plateau` or `-slow` to the name); no project checkout is needed.
CLI options include `--epochs`, `--seed`, `--width`, `--threads`,
`--learning-rate` and `--weight-decay`. Each export takes its optimizer defaults
from the selected width and preset, as every CPU/WASM engine does.

PEP 723 metadata installs pinned PyTorch from its CPU index and NumPy. On
reaching 99% test accuracy, an interactive run asks whether to continue with
the same optimizer. Noninteractive runs stop there; `--continue-after-target`
keeps them running to the epoch limit. The file recreates the experiment from
its seed. Browser weight/checkpoint downloads are removed. PyTorch's arithmetic
can shift convergence epochs relative to WASM or Rust/CUDA.

The canonical source is `python/rotate90.py`, with a committed uv lockfile.
`scripts/build-pytorch.ts` embeds it into the TypeScript exporter during build.
To validate the repository source and every exported architecture:

```sh
uv run --locked python/rotate90.py --epochs 1
node build/tests/pytorch-fixtures.js
uv run --locked tests/validate-pytorch-export.py
```

The validator checks exact initial tensors, all data and 21 minibatches, then
compares logits, every gradient and three AdamW updates with independent f64
fixtures for all five widths.

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
roles with software Canvas 2D in an OffscreenCanvas worker. The main thread
presents its completed pixel buffer with `putImageData` and marks a frame ready
only after that draw. Stale poses are acknowledged and discarded.
Magnitude uses `log1p(abs(weight) / initialRMS)`,
mapped to `strength / (1 + strength)`. E and A have separate initial RMS references
that stay fixed throughout training. Change ×4 uses one quarter of those references;
exactly unchanged weights are transparent. Nonzero changes use a single magnitude
factor for opacity, so small updates stay visible.
The legend and both views share the same color stops. Network uses
1,024 magnitude bands per sign. Its software WASM rasterizer accumulates every
line's scanline spans in an f32 difference buffer, then combines overlaps into
pixels. Color is a mass-weighted average; opacity is mass / (1 + mass).
Line mass scales with raster resolution so high-DPI output keeps its contrast.
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
bytes for W128 and 361,472 for W256. Adaptive training submissions target
50 ms; CPU workers yield between submissions for responsive pause. Metrics
export as CSV; PyTorch exports contain executable source and experiment defaults.

The startup dialog explains the task, inverse-based solution, eight checked
examples and the actual model. It includes a local rendered Mermaid SVG and
native model SVGs, tensor types and quantified equations. The ∑ button reopens
it. English, Spanish, German, Japanese, French and Russian are available.
Automatic selection follows browser preference order and regional tags;
manual choice persists locally. Language changes preserve training.
The Weights introduction explains the E, A and tied readout maps, their axes,
sign colors and fixed logarithmic scales, raw values and Change ×4, tooltips,
channel selection and Center R. Centering subtracts each channel's average
readout value over 97 answers and changes only the heatmap display. Introduction
text scrolls within its panel when needed, keeping navigation visible.
Next and Back traverse all seven introduction pages, the task, solution and
examples, then the model diagram, types and equations in sequence.

Translation sources are `scripts/translations.ts`. Model SVGs are generated
by `scripts/build-model-diagrams.ts`. Mermaid sources and hashes are in
`diagrams/`; regenerate them with `npm run build:diagrams` using an isolated
Chrome profile at `GROKKING_CDP_PORT`. Mermaid is not loaded at runtime. The
128-channel WAT is the source template in `wasm/`; the build specializes it
for the other four widths and emits SIMD/scalar binaries with private or shared memory.

The engine selector exposes 1, 2 or 4 CPU threads for both WASM variants.
Automatic selection uses four threads with at least six logical CPUs, two
with four or five, and one on smaller devices. Android and iOS automatically
use at most two training threads to leave capacity for the interface; explicit
thread selections remain available.
Parallel training partitions each batch, sums private gradients in fixed slot
order and performs one AdamW update on shared weights. Evaluation partitions
the complete split and combines f64 loss sums and integer correct counts.
Floating-point gradient regrouping can shift convergence epochs compared with
a single thread. Shared memory requires cross-origin isolation; the local
server supplies COOP/COEP headers. Without that capability the app uses one
thread. SIMD support is detected automatically, with scalar fallback.

## CPU optimization loop

After building, run a bounded search with exact output checks and repeated
timings. It tests input blocks, register tiles, vector unrolling, row-based
matrix multiplication and row groups separately for each width and CPU variant:

```sh
npm run optimize:cpu -- --rounds 2 --apply --output /tmp/grokking-cpu-optimization
npm test
npm run bench:cpu -- /tmp/grokking-cpu-benchmark
```

Candidates are assembled in memory and leave the live site assets untouched.
Each candidate must match every gradient for batches 1/37/48/512, the checkpoint
after five updates, probe logits and complete-split metrics byte-for-byte.
Promotion requires a confirmed 3% epoch-time gain without a training or
evaluation regression beyond the 3% timing allowance. `--apply` saves confirmed
choices in `scripts/cpu-config.ts`; rebuild and validate before using them.
Reports include all rejected trials, binary/source hashes and raw timings.
Without `--apply`, the loop writes reports and winner binaries only. W16 to W64
reuse the W128 choices; `--widths 256` limits the search to the listed
widths. A round with no confirmed gain stops the search early. Keep other training and rendering
idle during measurements; repeat on the target device before comparing results.

## CPU measurements (2026-10-15)

Node v24.21.0 on a Ryzen 5 5600H. All engines were measured sequentially,
with warmup and median timings.

Training and evaluation process equations in groups of four. Every matrix
value loaded from memory serves all four rows, and the softmax sums and
RMSNorm dot products of a group run side by side. Each output keeps its
single-row summation order and each gradient element receives the rows in
batch order, so results stay bit-identical. A group with an exact zero in a
normalized input or hidden vector takes the single-row kernels, which skip
zero inputs. The softmax maximum comes from SIMD lane maxima and a
first-index scan. ReLU uses a mask instead of WebAssembly's NaN-aware max,
which V8 compiles to long or branchy code, and the exponentials keep their
constants in registers. W256 walks the Dense
matrix in 16-input blocks: with 1,024-byte rows, longer column walks evict
each other from L1.

The search promoted four-row groups for every width and variant, 1.9–2.1×
faster per epoch, then 16-input blocks for W256, another 7%. An earlier
search without row groups had not confirmed those blocks. All 102 candidates
matched their references byte for byte. The rebuilt binaries for all five
widths also match the previous ones in 1,270 checks: gradients for batches
1–9, 37, 48 and 512, AdamW checkpoints, logits, metrics, partial evaluation
ranges, mixed exact zeros, an infinite weight, and the 2- and 4-thread
engines. NaN payloads count as equal because WebAssembly leaves them
unspecified.

| Width / variant | 1 thread: train / evaluate ms | 2 threads: train / evaluate ms | 4 threads: train / evaluate ms | Estimated epoch, 4 threads | Previous kernels, 4 threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / simd | 2.6 / 14.3 | 1.6 / 7.3 | 1.0 / 3.8 | 13.8 ms | 24.1 ms |
| 128 / scalar | 8.2 / 52.0 | 4.4 / 28.0 | 2.4 / 14.5 | 38.4 ms | 73.5 ms |
| 256 / simd | 7.1 / 43.5 | 4.0 / 22.5 | 2.5 / 11.1 | 35.8 ms | 70.8 ms |
| 256 / scalar | 25.0 / 151.6 | 13.0 / 79.2 | 7.0 / 40.9 | 110.6 ms | 225.7 ms |

Training timings cover one batch of 512, evaluation all 9,312 equations.
Epoch estimates use ten full batches plus evaluation, slightly overestimating
the actual 48-row final batch. Floating-point gradient regrouping changes
parallel checkpoint hashes. Every engine also reproduces the trained weight
hashes of the earlier benchmark. The four-thread W256 engine uses 3.5 MiB of
shared WASM memory, up from 3.4 MiB for the four row slots.

Complete software graph drawing at 1250×520 takes about 39 ms for W128 and
79 ms for W256, including weight classification and rasterization.
Highlighting a cached graph takes about 1 ms. Dense drawing runs in its own
worker and coalesces newer requests; the main-thread fallback remains available.
These full raster timings include work that older Canvas submission timings
did not wait for, so those historical numbers are not directly comparable.

[Raw timings of both kernel generations, both searches, the bitwise comparison
and drawing measurements](tests/results/cpu-optimization-2026-10-15/) include
binary and source hashes. The
[2026-10-14 measurements](tests/results/cpu-optimization-2026-10-14/) cover
W128 and the retired W1024 engine, whose four-thread SIMD variant took
123.0 / 257.7 ms per batch / evaluation and drew its graph in 642 ms.

## Validation and observed convergence

The TypeScript build and 89 Node tests pass: deterministic data and sampling,
strict grokking classification, native checkpoint hashes, scalar/SIMD logits,
all parameter gradients, shared E accumulation, finite differences and AdamW.
Two- and four-thread variants also pass gradient, AdamW and complete-split
evaluation checks for all five widths and batch sizes including 1, 48 and 512.
Scalar/SIMD raster tests verify actual pixels for all three E roles, Rotate90
signs, Dense connections, zero changes and offscreen clipping.
The original five-update single-thread CPU checkpoint, all gradients and probe
logits remain bit-for-bit identical; their pre-optimization hashes are checked
for both previously displayed widths and CPU variants.
The real UI trains W128 and W256 with GPU APIs forbidden,
pauses/resets, restores independent states, switches the W128 Fast, Plateau
and Slow presets, updates weights, and fits all three views. Localization
checks cover desktop/mobile dialog and workspace layouts, including Settings
and the complete scrollable weights and Probe contributions explanations.
Custom-setting checks validate the five supported widths, custom AdamW updates,
and four shared workers. The settings browser check follows
every introduction page forward and backward, trains W32 SIMD and W64 scalar,
checks reset, preset restoration and PyTorch defaults, and fits 30 localized
settings layouts with the Custom tab visible.
The mobile browser check uses Android Chrome emulation, DPR 3 and a sixfold CPU
slowdown. It switches every width and preset, rapidly changes models during
startup, repeatedly switches views at four screen sizes, checks bounded worker
counts, suspends running training and verifies bit-for-bit continuation. It also
checks that all six translated Dense/ReLU/skip/RMSNorm captions fit their
backgrounds.
The mobile memory regression trains W128 for 15 epochs, visits Network and
Weights, and checks 12 complete render/release cycles plus 120 interleaved view
switches. It verifies actual colored pixels in all three network connection
bands while simulating broken transferred-canvas presentation, checks recovery
after dropping stale frames, and bounds completed images in flight to one.
It compares every heatmap pixel, requires 2× resolution on the phone, holds
trainer shutdown pending to detect blank maps, and simulates canvas
restoration. Hidden surfaces must be 1×1, network raster memory must remain
below 4 MiB, and paused Weights must retain no dedicated workers. Returning to
Training must reproduce the uninterrupted epoch-16 weights bit-for-bit.
The Pages check loads all 20 packaged training modules, verifies the isolation
bootstrap without server headers, and trains every width with custom AdamW
parameters on four CPU threads under the `/grokking_abmod/` prefix.

The export integration check trains every width and preset and downloads their
actual `.py` files before and after learning. Sources remain unchanged by training.
With the fast preset, W128 on Chrome's four-thread WASM reaches 99.27% test
accuracy at epoch 30, automatically pauses, then continues to epoch 33 (99.96%)
with the same worker, optimizer and history. Reset restores epoch zero. The
downloaded W16 slow script also completes a CPU training epoch through uv
without the repository or server. The preset check trains the other eleven
presets in Chrome at seed 11 to their automatic pause and three more epochs.
Each pauses at the epoch of the Node column above, from W64 Fast at 21 to W128
Slow at 8,189. Four-thread wall time was 49 s for W128 Plateau, 99 s for W128
Slow and 267 s for W256 Slow, down from 88, 194 and 558 s before the row-group
kernels, which reproduced every pause epoch and accuracy.
[Export, pause, preset and layout evidence](tests/results/optimizer-presets-2026-10-15/)
records these checks separately from the historical convergence runs below. The
[2026-10-14 checks](tests/results/widths-export-2026-10-14/) used lr 0.001 and
weight decay 0.1.

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

Run the memory regression in a fresh profile so dedicated-worker counts include
only that run. Its resource limits and exact continuation were also checked
with graphics compositing enabled and a 48 MiB V8 heap limit.

```sh
GROKKING_CDP_PORT=9224 npm run test:browser
GROKKING_CDP_PORT=9224 npm run test:settings-browser
GROKKING_CDP_PORT=9224 npm run test:help-browser
GROKKING_CDP_PORT=9224 npm run test:mobile-browser
GROKKING_CDP_PORT=9224 npm run test:mobile-memory
GROKKING_CDP_PORT=9224 GROKKING_PREVIEW_ORIGIN=http://127.0.0.1:8081/grokking_abmod npm run test:pages-browser
GROKKING_CDP_PORT=9224 npm run test:export-browser
GROKKING_CDP_PORT=9224 node build/tests/optimizer-browser.js
GROKKING_CDP_PORT=9224 npm run test:rendering
GROKKING_CDP_PORT=9224 npm run test:i18n
GROKKING_CDP_PORT=9224 node build/tests/measure-topology.js
node build/scripts/measure-cpu.js /tmp/grokking-cpu-benchmark
node build/scripts/measure-wasm.js dense-rotary simd tests/results/new-wasm-run 11 slow
```

`measure-wasm` takes an optional seed and preset (`fast` by default).
