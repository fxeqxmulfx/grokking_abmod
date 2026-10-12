# Rotate90 Grokking Lab

**Current mode: local only.** Run the preview below. Training executes entirely
in the browser; Node serves static files on localhost. The previous Sites
identity remains in the ignored `.openai/hosting.json` for cleanup. Connected
Sites tools currently provide no deletion or unpublication operation, so the
previous hosted copy has not been confirmed removed. Do not publish updates
without a new user request.

A static frontend trains division-modulo-97 models from scratch and displays
live loss, accuracy, weights, and probe activations. Both tabs use **one shared-E
Rotate90 Square Neighbor residual architecture**, differing only in width:

| Tab | Channels | Unique parameters | Five-seed Rust result |
| --- | ---: | ---: | --- |
| Rotate90 · 128 | 128 | 13,952 | Fast generalization; no memorization plateau |
| Rotate90 · 1024 | 1024 | 111,616 | Plateau plus rapid held-out rise on 5/5 seeds |

See [the replicated Rust and browser study](../../docs/rotary-neighbor-results.md).
Hardware WebGPU is preferred, with handwritten WASM SIMD128 and scalar WASM
fallbacks. No training backend, model download, Python runtime, or JavaScript
ML framework is required. Node v24.21.0 was used for checks; `.nvmrc` selects it.

```sh
node web/own-channel/server.mjs
```

Open `http://127.0.0.1:8080`. The dashboard fits the viewport without page
scrolling. Narrow and short windows have Training, Weights, and Channels views.
Each model retains its own weights, AdamW moments, sampler, and curves;
switching tabs pauses the previous experiment. Changing engine or seed resets
the selected model. Switching views preserves training state.

## Shared architecture

One `97 × W` table supplies both operands and the readout. Each adjacent
channel pair in operand `b` undergoes a fixed +90° rotation:

```text
J(u, v) = (-v, u)
x0 = E[a] + J(E[b])

# Two independent residual blocks:
r = RMSNorm(x)
z1 = M1(r)
z2 = M2(z1²)
x_next = x + M3(z2²)

logits = RMSNorm(x2) Eᵀ
M(z)[c] = α[c] z[c] + β[c] z[(c + 1) mod W]
```

The fixed role rotation draws on [RoFormer's pairwise rotations](https://arxiv.org/html/2104.09864v5).
Its angle depends only on operand role. There is no attention or residue-value
encoding through rotation. All six mesh layers have independent signed self
and neighbor weights. RMSNorm uses all channels, epsilon 1e-5, and no learned
affine parameters. All layers are bias-free. Unique parameters total
`97W + 6 × 2W = 109W`.

Input roles and readout contribute to the same gradient for `E`, with one
AdamW state per unique parameter. Initialization uses SplitMix64, `E` uniform
in `[-1/√97, +1/√97]`, and identity mesh edges (`α=1, β=0`). With Square, an
initial branch computes `RMSNorm(x)⁴`, which the skip adds to `x`.

The selected channel's ancestors form a **4 → 3 → 2 → 1** diagram: ten nodes,
twelve learned edges, and the skip connection. Nodes show actual probe
activations and edges show signed weights. RMSNorm also depends on channels
outside this local diagram. The block selector exposes all six mesh layers.
The IO atlas shows shared `E`, derived `J(E)` for operand `b`, and tied `Eᵀ`.

Click nodes, heatmaps, or the channel contribution plot to select a channel;
the numeric selector reaches any channel, including the final wraparound
channel 127 or 1023. When multiple channels share one plot pixel, the plot
shows their maximum contribution and selects its actual channel. Weight,
change-from-initialization, and centered-readout views remain available.

## Protocol and measurements

Both tabs default to seed 11: 9,312 division pairs, a fixed 50% split, 4,656
pairs per split, nine batches of 512 plus an exact 48-row tail per epoch,
AdamW learning rate 0.001, decay 0.1, betas (0.9, 0.98), epsilon 1e-8, ten
warmup updates, and a 10,000-epoch cap. Both complete splits are evaluated
**every epoch**. The step button advances one epoch. Optional stopping uses
first held-out accuracy ≥99%. Curves contain actual observations.

SplitMix64 initialization, dataset, and shuffle order follow Rust. Initial
weights match Rust byte for byte at both widths for seeds 7 and 11. Reduction
order and WASM softmax approximations differ from CUDA, so crossing epochs can
differ between engines.

| Width / seed 11 | Rust test ≥99% | WASM SIMD test ≥99% | WebGPU test ≥99% | WASM training + evaluation | WebGPU training + evaluation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 133 epochs | 133 epochs | 133 epochs | 9.32 s | 1.73 s |
| 1024 | 187 epochs | 187 epochs | 186 epochs | 95.09 s | 13.40 s |

W1024 retains a **34-epoch train =100% / held-out ≤10% interval** (22–56)
in all three engines. Its train ≥99% plateau lasts 41 epochs (15–56), and its
held-out ≤10% → ≥90% rise takes 80 epochs on Rust and WASM, 79 on WebGPU.
W128 has no qualifying memorization plateau and rises over 49 epochs.
Gray chart shading marks the ≥99%-train interval; violet marks perfect train.

The grokking label requires at least 25 contiguous epochs of train ≥99% /
held-out ≤10%, then a ≤250-epoch rise to ≥90%, followed by ≥99%. Train dips or
missing observations break the plateau. Later test dips preserve the first
90% crossing, and later streaks preserve the first confirmed plateau. The
compact model reports `Generalized`; the wider model reports
`Short-plateau grokking` when the criterion is met.

CPU timings use one thread on an AMD Ryzen 5 5600H in Node v24.21.0. WebGPU
measurements use Chrome 155.0.8059.39 and a hardware adapter (`fallback: false`)
whose description is hidden. Times exclude engine setup and normal live weight
readbacks. Raw histories, source hashes, CSV exports, and real UI training
outcomes are preserved in [`tests/results/`](tests/results/). Earlier
architectures and measurements remain in
[the historical comparison](../../docs/grokking-browser-models.md).

## Efficient execution

- Rotate90 uses paired shuffles and signs, with no trigonometry or learned
  input projection. Readout uses the same embedding table.
- WebGPU keeps parameters, activations, gradients, and optimizer state on the
  device. Deterministic reductions avoid floating atomics, readout gradients
  use tiled matrix products, and the three embedding roles merge before AdamW.
- Only 512 equations' intermediate states are cached: 4.25 MiB at W128 and
  34 MiB at W1024. Complete-split evaluation uses workgroup-local branch states.
  Wide shaders use 128 threads with eight channels each and less than 16 KiB
  of workgroup storage, within default device limits.
- Up to 32 updates share a command buffer. Submission size adapts toward 50 ms;
  CPU workers yield through a MessageChannel so Pause remains responsive.
- Full-split metrics are reduced to a small result. Weight snapshots cost
  55,808 bytes at W128 or 446,464 at W1024. Default 5 Hz readbacks transfer
  about 0.28 or 2.23 MB/s; 1 Hz reduces synchronization overhead.
- Hidden views do not draw. Heatmaps use pixel buffers, and probe inference
  uses the latest actual weight snapshot.

The visuals follow the weight/activation diagrams in Stephen Wolfram's
[2024 minimal-models article](https://writings.stephenwolfram.com/2024/08/whats-really-going-on-in-machine-learning-some-minimal-models/).
Readout centering removes the class-independent component ignored by softmax.

## Handwritten WebAssembly and checks

`wasm/neighbor-rotary.wat` and `wasm/neighbor-rotary-wide.wat` implement the
current forward/backward and optimizer kernels. Shared scalar/SIMD matrix and
rotation routines are inserted at build time. Earlier own-channel, concat,
untied Neighbor, and dense residual trainers remain as reference baselines.
No WASM module imports, allocator, threads, shared memory, memory growth, or
cross-origin isolation headers are needed.

| Width | SIMD binary | Scalar binary | Fixed linear memory |
| --- | ---: | ---: | ---: |
| 128 | 3,723 bytes | 3,263 bytes | 448 KiB |
| 1024 | 3,753 bytes | 3,293 bytes | 2.69 MiB |

Binaries are included in `dist/`. WABT is the only build dependency and is not
shipped to the browser. Rebuild and test from this directory:

```sh
nvm use
npm ci --ignore-scripts
npm run build:wasm
npm test
```

The complete Node suite passed 73 tests; an added four-checkpoint parity test
also passed separately. Tests cover initialization, data and epoch sampling,
finite differences, every parameter gradient for batches 1/37/48/512, shared
embedding accumulation, residual RMSNorm/skip derivatives, signed Square
inputs, circular edges, scalar/SIMD AdamW, and grokking classification. Both
current WebGPU kernels passed 18 f64 gradient/optimizer checks each, including
three AdamW updates. Fresh complete runs reached the reported targets.

From the repository root, use a temporary Chrome profile with CDP port 9224:

```sh
GROKKING_PREVIEW_ORIGIN=http://127.0.0.1:8080 \
  node web/own-channel/tests/browser.mjs
```

UI checks cover both models, pause/resume, reset, independent state, curves,
live weights, channel and block selection, and viewport fit from 320×568 to
1920×1080, including tablet and landscape layouts.

Downloaded weights are raw little-endian f32: shared `E`, then six mesh
layers' interleaved self/neighbor weights. Metrics export as CSV. Weight
exports omit AdamW state and cannot resume an optimizer run.
