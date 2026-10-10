# CUDA profiling and training performance

The grokking implementation was committed as `0bdc869` before changing CUDA
kernels. Its full mod-97 epoch and CPU/CUDA numerical checks passed before
profiling. The measurements below use an RTX 3050 Laptop GPU (16 SMs, `sm_86`),
driver 615.71.09, CUDA 13.4, Nsight Systems 2026.3.2, and Nsight Compute 2026.3.1.
Weights, activations, gradients, reductions, and AdamW remain FP32. The kernels
do not use BF16, TF32, or Tensor Core matrix operations.

## Reproducible training workload

`mesh-gpt-bench` measures the actual safe training API with answer-only loss,
width 128, four heads, two blocks, context four, a 99-token vocabulary, and
512 fixed training equations (2048 rows). Model initialization and inputs use
seed seven. AdamW uses learning rate `0.001`, decay `1`, and betas `(0.9, 0.98)`.
GPU warmup uses the same constant rate; it serves timing rather than the paper's
learning-rate schedule.

```bash
cargo oxide run --bin mesh-gpt-bench -- \
  --steps 30 --warmup 10 --trials 3 --output target/mesh-gpt-bench
```

Each trial starts from identical weights, resets AdamW, runs ten untimed updates,
and synchronizes before timing. Wall-clock timing covers 30 training updates,
including host validation, token uploads, loss downloads, and completion of the
last AdamW update. The final parameter download is included. Full-split evaluation,
checkpoint writing, and CSV output happen outside the timed interval. Workspace
allocation and CUDA compilation happen before timing.

| Implementation | Trial 1 (ms/update) | Trial 2 | Trial 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Committed baseline kernels | 160.700 | 160.782 | 160.757 | 160.757 |
| Token/channel embedding gradients | 53.457 | 53.334 | 53.332 | 53.334 |
| Embedding gradients and chunked mesh reduction | 18.362 | 18.146 | 18.141 | 18.146 |
| Parallel mesh forward | 13.465 | 13.167 | 13.155 | 13.167 |
| Parallel mesh input gradients | 10.403 | 10.107 | 10.103 | 10.107 |
| Mesh gradient chunks of 32 rows | 9.804 | 9.007 | 9.000 | 9.007 |
| Transposed embedding table for the head | 9.052 | 8.303 | 8.312 | 8.312 |
| Parallel RoPE, attention backward, and copies | 8.091 | 7.398 | 7.402 | 7.402 |
| Final const-generic kernels and safe module loading | 7.507 | 7.253 | 7.256 | 7.256 |
| Cooperative FP32 RMSNorm and QK-Norm | 6.628 | 6.454 | 6.455 | 6.455 |
| Warp mesh-gradient reduction | 6.386 | 6.159 | 6.165 | 6.165 |
| Chunked FP32 tied-embedding gradients | 6.517 | 5.749 | 5.759 | 5.759 |

The const-generic median is **22.16 times faster** on this workload, and **2.50 times
faster** than the first chunked implementation. GPU frequencies were
not locked; these are local measurements rather than a throughput guarantee for
other shapes or devices. `timings.csv`, `run.txt`, and the first trial's `model.bin`
make the timing setup and numerical result inspectable.

The cooperative normalization path improves the previous median by **11.0%**.
A fresh run of the saved previous binary measured `7.794`, `7.319`, and `7.316`
ms/update, with median `7.319`; the corresponding reduction is **11.8%**.
The additional gradient reductions bring the median to **5.759 ms/update**,
**20.6% lower** than the earlier `7.256` ms implementation and **27.91 times
faster** than the original baseline.

## Profile and changes

A focused Nsight Systems trace captures five warmup and twenty timed updates
without full-split evaluation. Before optimization, embedding backpropagation
accounted for 67.2% of kernel time and mesh weight gradients for 23.8%.

| Kernel family | Baseline GPU ms/update | Const-generic GPU ms/update |
| --- | ---: | ---: |
| Embedding gradient and conversion to feature rows | 108.819 | 1.099 |
| All mesh weight-gradient kernels | 38.473 | 1.798 |
| Mesh forward and input gradients | 6.433 | 1.455 |
| Vocabulary head forward/backward and table transpose | 2.742 | 0.331 |
| Attention input gradients | 0.788 | 0.096 |
| Residual additions and copies | 1.722 | 0.359 |

These numbers divide each family's summed kernel duration by all 25 captured
updates. They exclude host overhead. They come from separate profiled runs from
the unprofiled wall-clock benchmark above.

Originally, one embedding thread owned a token vector and serially traversed
every channel and every input row. The new kernel gives each token/channel pair
its own scalar output, followed by a conversion to the feature-row format used
by mesh backpropagation. For this shape the grid grows from one block to 99.
Each contribution is still accumulated in row order. Nsight Compute measured
SM throughput `0.28% -> 26.58%` and achieved occupancy `7.08% -> 51.56%` in the
isolated embedding-kernel captures.

The original mesh weight-gradient kernel assigned one thread per edge, leaving
each thread to sum 2048 rows. The new path splits rows into chunks of 32,
computes partial sums for each chunk/edge pair, then reduces those sums per edge.
The partial-sum kernel launches 128 blocks for the default batch. Batches with at
most 128 rows and the smaller embedding-mesh reduction use the direct path.
Chunk storage is reused across mesh layers within a workspace.

Mesh forward/input gradients, RoPE, head input gradients, attention input
gradients, copies, and residual additions give each output channel its own
thread. Adjacent threads access adjacent feature channels. The vocabulary head
reads a temporary transposed embedding table so adjacent token outputs read
adjacent values while retaining the original channel accumulation order.

Both paths use typed `DisjointSlice` writes and checked launch contracts, with
exclusive output ownership, no atomics, and no additional `unsafe` operations.
The contracts validate chunk counts, row extents, scalar-output lengths, and
parameter offsets. Existing `MSHGPT07` checkpoints use the same model equations
and parameter layout.

All tensor capacities are Rust const parameters. The final benchmark uses
`SpecializedMeshGpt<128, 256, 128, 8>` with active dimensions from the workload
above. Other capacities compile the same kernel definitions; active dimensions,
layer counts, and mesh depths are checked runtime values. See
[Mesh GPT](mesh-gpt.md#gpu-execution-and-verification) for a wider specialization.
The CUDA Rust patches are published on `experiments`, and the project's Git
dependencies and CUDA compiler backend use revision
[`9a905694`](https://github.com/fxeqxmulfx/cuda-rust/commit/9a9056948298e550de137b0bb303390f4be9185a).
Generic kernel identity and embedded artifact provenance now also allow safe
`kernels::load(&context)?` in the GPT constructor.

The earlier RMSNorm and QK-Norm optimization assigned one thread per output channel when the capacity
and active head width were multiples of 32. Each complete warp cooperatively
reduced squared inputs or the backward dot product. Smaller and unaligned
shapes used the original serial reduction. The cache stored each channel's
normalized value beside its inverse RMS so typed disjoint writes remained safe.
Checked launch contracts reject incomplete head geometry and overflowing
dimensions; the forward and backward paths retain FP32 arithmetic.
The new 25-update trace measures normalization forward/backward together at
`0.539` GPU ms/update, down from `1.706` in the previous capture.

The final mesh-gradient reduction now uses a complete warp per edge. Lanes
share the chunk sums, and only lane zero can obtain the typed `WarpIndex`
witness needed to write that edge. Entire warps skip neighboring parameter
ranges, preserving full participation in the reduction. The latest trace
measures this kernel at `0.145` GPU ms/update, down from `0.545` after the
normalization optimization.

Tied-embedding gradients use independent chunks of 128 rows for every batch
and model width. Each partial combines the repeated input
lookup and dense output-head contributions in FP32. The second kernel reduces
directly into feature-row storage, avoiding a separate serial conversion.
Small batches use one chunk through the same checked kernels. The chunk length is a Rust const
parameter, while vocabulary and width remain validated runtime dimensions.
The latest trace measures partials plus reduction at `0.579` GPU ms/update,
versus `1.087` for embedding backward plus conversion in the preceding trace.
The default workload reuses a `0.77 MiB` partial-sum allocation across updates.
All new writes use typed disjoint ownership and checked launch contracts;
`src/gpt/gpu.rs` contains no `unsafe` operations.

| Representative kernel | Blocks | Registers/thread | SM throughput (%) | Achieved occupancy (%) |
| --- | ---: | ---: | ---: | ---: |
| Direct embedding gradients (previous) | 99 | 23 | 26.58 | 51.56 |
| Mesh gradient partial sums | 128 | 26 | 30.79 | 64.55 |
| Decoder mesh forward | 2048 | 18 | 54.45 | 79.93 |
| Chunked embedding gradients | 1584 | 28 | 53.73 | 96.97 |
| Warp mesh-gradient reduction | 384 | 20 | 9.99 | 54.50 |

Each capture uses 128 threads per block and an initial update with unlocked
frequencies. The decoder capture skips the four embedding-mesh launches.
The last two captures use the selected FP32 kernels after the normalization
and gradient optimizations. Their durations are `566.82` and `4.77` microseconds;
the reduction capture measures one layer, while the traces sum all layers.
Occupancy and SM throughput measure different things; these counters do not
establish that the whole training workload reaches 80% of GPU peak throughput.
The latest FP32 trace's largest remaining families are mesh forward/input
gradients (`1.467` GPU ms/update), mesh weight gradients (`1.400`), and embedding
gradient reductions (`0.579`). Normalization forward/backward is now `0.543`.
Raw timings are saved in `fp32-reductions-stats_cuda_gpu_kern_sum.csv` alongside
the earlier captures and numerical comparisons.

## Capture profiles

Build first so compilation does not enter the capture:

```bash
cargo oxide build --arch sm_86 -- --release --locked --bin mesh-gpt-bench
mkdir -p target/mesh-gpt-profile

nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=target/mesh-gpt-profile/training \
  target/release/mesh-gpt-bench --steps 20 --warmup 5 --trials 1 \
  --output target/mesh-gpt-profile/training-run

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  --output target/mesh-gpt-profile/training-stats \
  target/mesh-gpt-profile/training.nsys-rep

ncu --section SpeedOfLight --section LaunchStats --section Occupancy \
  --kernel-name 'regex:gpt_embedding_gradient_partials_TID_.*' \
  --launch-count 1 --clock-control none \
  --force-overwrite --export target/mesh-gpt-profile/embedding-counters \
  target/release/mesh-gpt-bench --steps 1 --warmup 0 --trials 1 \
  --output target/mesh-gpt-profile/counters-run

ncu --import target/mesh-gpt-profile/embedding-counters.ncu-repz --page details
```

Use `--kernel-name 'regex:gpt_mesh_gradient_partials_TID_.*'` for the chunked
mesh kernel, or `--kernel-name 'regex:gpt_mesh_TID_.*' --launch-skip 4` for the
first decoder mesh forward kernel. Generic entry names include a compiler
function-item type ID, so the regex avoids depending on a particular hash.
Nsight Compute replays kernels to collect counters, so its displayed benchmark
wall time must not be used for speed comparisons. Its captures above use one
initial update and unlocked frequencies; the warmed Nsight Systems trace and
unprofiled trials supply the performance comparison.

Performance-counter access must be enabled by the driver administrator. R610+
drivers support granting the `profiler-device` capability to a specific user,
without allowing profiling for every account. Follow
[NVIDIA's counter-permission instructions](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
and use the minor number from `/proc/driver/nvidia-caps/sys-minors`.
The current session successfully captured both baseline and optimized counters.
Raw profiles, exported statistics, saved binaries, and `comparison.json` are in
the ignored `target/mesh-gpt-profile/` directory.

## Numerical validation

CUDA verification checks direct and chunked gradient sums against f64 at widths
2, 4, and 128 with 1, 31, 32, 33, 127, 128, 129, 257, 2048, and 4096 rows. It covers linear
and ELU layers, untouched neighboring parameter ranges, incorrect chunk counts,
and short output buffers. Full-model checks compare every parameter gradient,
masked and dense losses, AdamW continuation/reset, and context-dependent learning
against independent CPU autodiff.

Full-model CUDA checks also cover compact and non-power-of-two capacities,
width 256, 16 heads, vocabulary 257, context 129, and three blocks. They compare
every gradient and AdamW update against independent f64 autodiff and reject
configurations exceeding each capacity. Context-dependent learning reduces
cross-entropy from `1.98213` to `0.00153` and predicts every training target.

The embedding-only optimization produced a byte-identical checkpoint after 40
updates. The final implementation's maximum parameter difference from the
baseline after the same 40 updates is `4.17e-7`, with relative parameter L2
difference `4.22e-8`. Chunked sums change the order of f32 additions.

A final mod-97 run completes all 4704 training examples in one epoch, including
the 96-example tail. Training cross-entropy decreases from `7.419670094` to
`6.532873894`, and validation cross-entropy from `7.432581042` to `6.547292569`.
Its maximum parameter difference from the pre-optimization epoch checkpoint is
`5.96e-8`. This verifies learning and numerical agreement; it does not establish
grokking. All checks retain the model equations and FP32 precision.

Cooperative normalization verification additionally covers head widths 32, 64,
and 128, zero and tiny inputs, multiple heads, and invalid launch geometry.
All CUDA comparisons and 27 CPU tests pass, as does strict Clippy. After the
40-update benchmark, its maximum parameter difference from the previous FP32
binary is `1.91e-6` (relative L2 `1.08e-7`). A complete mod-97 epoch again covers
all 4704 examples; train/validation losses reach `6.532873886`/`6.547292552`,
with maximum parameter difference `2.32e-7` from the previous epoch checkpoint.
The small differences reflect the warp reduction's addition order.

The embedding-gradient verifier independently sums both contributions in f64
at widths 2, 18, 32, and 128, vocabularies 3, 17, and 99, and up to 4096 rows.
It checks chunk tails, untouched padded channels, malformed chunk counts,
overflowing dimensions, and short buffers. The complete CUDA suite, 27 CPU tests,
and strict Clippy pass after both gradient optimizations. The 40-update maximum
parameter difference from the previous const-generic FP32 binary is `1.43e-6`
(relative L2 `8.32e-8`). A complete 4704-example epoch reaches train/validation
losses `6.532873915`/`6.547292586`; its maximum parameter difference from the
previous epoch checkpoint is `2.44e-7`. All storage and arithmetic remain FP32.

Embedding chunks of 32 and 64 rows were also compared with the final 128-row
choice. Their medians were `5.749` and `5.737` ms/update, respectively, versus
`5.734` for a repeated 128-row run. The smaller chunks do not demonstrate an
improvement and require four or two times as much partial-sum storage, so the
implementation keeps 128. Verification covers boundaries around 32, 64, and
128 rows even though the selected chunk size is 128.

## Compiler flags

Rust's release profile already uses `opt-level=3`; see the
[Cargo release defaults](https://doc.rust-lang.org/cargo/reference/profiles.html#release).
The CUDA architecture is explicitly `sm_86`. CPU `target-cpu=native` selects
instructions for the host processor; it does not select a CUDA architecture.

The 128-row embedding-chunk implementation was rebuilt and measured with the
same 30-update, ten-warmup, three-trial protocol:

| Compiler setup | Trial 1 (ms/update) | Trial 2 | Trial 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Portable CPU release, default CUDA LLVM pipeline | 5.989 | 5.722 | 5.734 | 5.734 |
| CPU `target-cpu=native`, default CUDA LLVM pipeline | 6.402 | 5.729 | 5.726 | 5.729 |
| CPU native, CUDA LLVM `opt` and `llc` at O3 | 6.494 | 5.735 | 5.730 | 5.735 |

The CUDA Rust PTX pipeline runs LLVM's middle-end at O2 and uses `llc`'s default
optimization level. The O3 experiment temporarily replaced `-O2` and
`default<O2>` with O3 and supplied `llc -O3` through process-local tool overrides.
Both experimental builds pass CUDA verification and produce byte-identical
40-update checkpoints to the portable build. Differences between warmed
medians are within `0.1%`; these flags provide no demonstrated speedup here.
Native and O3 overrides are not persisted in the repository configuration.

The CPU-native build on this AMD Ryzen 5 5600H can be reproduced in a separate
output directory:

```bash
RUSTFLAGS='-C target-cpu=native' CARGO_TARGET_DIR=target/native-release \
  cargo oxide build --arch sm_86 -- --release --locked \
  --bin mesh-gpt --bin mesh-gpt-bench

target/native-release/release/mesh-gpt-bench \
  --steps 30 --warmup 10 --trials 3 \
  --output target/mesh-gpt-profile/native-release-run
```

## Code simplification

Ordinary functions and closures replace repeated launch checks, buffer casts,
CLI parsing, and checkpoint decoding. The independent f64 reference shares its
own repeated operations. A single checked chunked embedding-gradient pipeline
now handles every batch and width, removing two unused kernels and an allocation.
No new Rust macros or dependencies are introduced. Rust source decreases from
7483 to 7137 lines (346 fewer), including 286 fewer lines in the CUDA GPT module.

The same 50-update, ten-warmup, three-trial protocol compares saved binaries:

| Workload | Previous median (ms/update) | Simplified median |
| --- | ---: | ---: |
| Width 128, batch 512 | 5.731 | 5.716 |
| Width 8, two heads, batch 512 | 3.220 | 2.583 |
| Width 128, batch 1 | 1.048 | 0.965 |

The narrow and single-equation improvements come from the unified embedding
reduction. Default and single-equation checkpoints are byte-identical to the
previous binary; the narrow workload differs by at most `3.58e-7` (relative L2
`6.16e-8`). All CPU tests, strict Clippy, both CUDA verification suites, invalid
launch checks, and the complete mod-97 epoch pass.

## Current optimization criteria

Continue measured FP32 optimization while reducing unnecessary code and GPU
memory, using ordinary Rust and const generics without new macros. Memory
savings are an independently useful result even when throughput is unchanged.
Report both memory saved and timing tradeoffs, and use the freed capacity for
larger batches, wider/deeper models, and longer contexts. Retain fixed-shape
comparisons separately from stress shapes adjusted to use about 3 GiB. Preserve
the architecture, numerical checks, safe client APIs, and English repository
content; run Python through uv and commit validated progress frequently.

## Large-memory and long-context workloads

The benchmark accepts `--width`, `--heads`, `--layers`, `--attention-depth`,
`--ffn-depth`, `--context`, and `--batch`. Widths through 32, 128, 256, 512, and 1024
select matching const-generic capacities at context up to 128. Widths through
128 additionally support context capacities 1024 and 4096. All specializations
use vocabulary capacity 128 for the fixed 99-symbol dataset, head capacity four
or eight, and context capacity four for sequences through four tokens. Batch validation permits
at most 131072 token rows; mesh-gradient contracts cover all 4096 row chunks.

Run the original workload and five current stress scenarios near 3 GiB of GPU memory:

```bash
cargo oxide build --arch sm_86 -- --release --locked \
  --bin mesh-gpt --bin mesh-gpt-bench
uv run --locked scripts/bench-cuda.py \
  --steps 20 --warmup 5 --trials 3 \
  --output target/mesh-gpt-profile/shapes
```

The runner measures per-process memory with `nvidia-smi` in a separate probe,
then times training without telemetry or profiler activity. `summary.csv`
records the median time, sampled peak process memory, and bytes owned by device
buffers. The process measurement also includes allocator, context, and module
overhead. Each directory retains timings, metadata, memory samples, logs, and
the first trial's checkpoint. Allocator rounding can make the same logical
allocation use different physical memory at different batch sizes.

At context four, inputs are mod-97 equations with answer-only loss. Batches
larger than the training split repeat its indices cyclically. Other contexts
use complete cyclic symbol sequences with dense next-token targets; both
attention and its backward pass process the entire sequence. This is a
reproducible profiling workload, not an evaluation of language-model quality.

CPU boundary checks and independent f64 CUDA sums cover the new 65536-row
limit, a 65535-row tail, and the transition past 4096 rows. Invalid counts,
overflowing dimensions, and short buffers remain rejected before launch.

### Cooperative XSA

The first expanded measurements use five warmup updates and ten timed updates
in each of three trials. Memory comes from separate probes. Equations repeat
only in the large-batch workload; the other short-context batches fit the split.

| Scenario | Width / blocks / batch / context | Process memory (MiB) | Before (ms/update) | Cooperative XSA |
| --- | --- | ---: | ---: | ---: |
| Original | 128 / 2 / 512 / 4 | 256 | 5.731 | 5.488 |
| Large batch | 128 / 2 / 8192 / 4 | 3002 | 148.798 | 111.254 |
| Wide model | 1024 / 2 / 1152 / 4 | 2932 | 131.836 | 108.554 |
| Deep model | 256 / 20 / 512 / 4 | 3152 | 93.636 | 88.006 |
| Combined | 512 / 4 / 1200 / 4 | 3178 | 116.978 | 89.032 |
| Long context | 128 / 2 / 16 / 1024 | 2882 | 1158.725 | 1142.346 |

XSA now owns scalar outputs per channel. Complete aligned warps reduce head
norms and projections; other shapes perform serial head reductions. Both
directions share one ordinary device function. Row/head alignment and bounds
are checked before any full-warp collective, including the unaligned fallback.
The change adds no workspace allocation and keeps FP32 and the XSA norm floor.

In the large-batch trace, XSA forward/backward together decrease from `39.584`
to `1.684` GPU ms/update. An isolated backward capture decreases from `12.49`
ms to `467.97` microseconds, with SM throughput `0.68% -> 60.96%` and DRAM
throughput `48.71% -> 93.39%`. The bandwidth number describes this kernel.
The long-context trace instead attributes 56.4% to attention forward and 33.0%
to score backward, making those the next targets.

The XSA verifier covers serial and cooperative heads, widths through 128,
head widths 32/64/96/128, zero/tiny values, padding, malformed geometry, and
short outputs. Full-model f64 checks now include width 1024 with 256-channel
heads. Losses and gradients retain the `5e-4 * (1 + abs(reference))` tolerance.
Optimizer checks compare scalar CPU FP32 AdamW with the exact GPU gradients it
consumes, at the tighter `1e-6 * (1 + abs(reference))` tolerance; gradients are
also checked against f64 at the actual weights before each update. Both the
original and cooperative width-1024 kernels showed amplified tiny-gradient
rounding when the previous test compared AdamW updates driven by separate f64
and f32 gradients. The new check isolates gradient accuracy from optimizer
accuracy without relaxing either requirement.

After 15 updates, maximum parameter differences from the saved original range
from `6.56e-7` to `1.80e-3`; the widest model's relative L2 difference is
`6.07e-5`. Default and large-batch differences stay below `2.99e-6`. All 28 CPU
tests, strict Clippy, full CUDA verification, and the complete mod-97 epoch
pass. Profiles and comparisons are retained under `target/mesh-gpt-profile/`.

### Warp-owned attention rows

Attention forward and score backward now share a cooperative FP32 dot product.
One complete warp owns each output row through `WarpIndex`; only its owning
lane obtains mutable references, while all lanes reduce channels and gather
values. Broadcasts distribute probability values and collected output channels.
The same definitions handle small, unaligned, and wide heads and introduce no
new buffers or shape-specific kernel variants.

| Scenario | After XSA (ms/update) | After warp attention |
| --- | ---: | ---: |
| Original | 5.488 | 5.340 |
| Large batch | 111.254 | 79.918 |
| Wide model | 108.554 | 103.801 |
| Deep model | 88.006 | 83.245 |
| Combined | 89.032 | 84.836 |
| Long context | 1142.346 | 385.169 |

The long-context trace measures attention forward at `180.978` GPU ms/update
and score backward at `98.760`, down from `649.363` and `379.682` before these
changes. Input-gradient gathering remains `67.925` ms/update. An isolated
forward capture takes `87.90` ms and reaches `80.02%` SM throughput but `3.89%`
DRAM throughput. Instruction count and partially active warp paths now matter
more than external memory bandwidth for this kernel.

Full f64 logits/gradient checks, strict FP32 optimizer checks, all 28 CPU tests,
and the complete mod-97 epoch pass. The context-learning check reaches
cross-entropy `0.00145` and predicts every target. The long-context checkpoint
differs from the preceding implementation by at most `5.96e-8` after 15 updates
(relative L2 `9.94e-9`); the widest case differs by `4.02e-4` (relative L2
`1.70e-5`). The saved binaries and all six checkpoint comparisons remain in
the ignored profile directory.

### Shared normalization statistics

RMSNorm now uses two kernel definitions for all shapes: one warp per head
computes statistics, then scalar channel owners apply normalization. Backward
recomputes the inverse RMS from retained graph inputs and reads normalized
features already in the graph. A single scratch allocation of two FP32 values
per row/head replaces every per-node cache. This removes duplicate normalized
features, repeated inverse RMS values, two kernel definitions, cache IDs in the
execution plan, and 101 Rust lines overall.

| Scenario | Before (ms/update) | Shared statistics | Before memory (MiB) | After memory (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Original | 5.337 | 5.380 | 256 | 238 |
| Large batch | 80.071 | 80.466 | 3002 | 2714 |
| Wide model | 104.131 | 83.095 | 2932 | 2608 |
| Deep model | 83.652 | 81.686 | 3152 | 2828 |
| Combined | 85.020 | 76.854 | 3178 | 2838 |
| Long context | 378.799 | 377.517 | 2882 | 2738 |

These are the same fixed shapes, five warmup updates, ten timed updates, and
three trials, with separate process-memory probes. Wide-model update time
decreases by 20.2%, and the combined scenario by 9.6%; the original and
large-batch timing differences are under 1%. Large scenarios free 144--340 MiB
of actual process memory. The wide-model Nsight Systems trace measures both
normalization kernels together at 8.546 GPU ms/update, compared with 30.749
for normalization forward/backward in the initial wide-model trace.
All six 15-update checkpoints are byte-identical to
the preceding implementation. All 28 CPU tests, strict Clippy, full f64 CUDA
verification, context learning, and the complete mod-97 epoch pass.

The additional launch costs matter for small shapes. Separate 30-update,
ten-warmup, three-trial measurements show width eight increasing from 2.480 to
2.639 ms/update and batch one from 0.682 to 0.729. This is an explicit timing
tradeoff for the memory saving; reducing those extra launches remains useful.

An earlier single-warp-row normalization experiment saved only each head's
inverse but made lane zero gather and store whole rows. It removed 112 Rust
lines and saved memory, yet increased original/large-batch/deep/combined times
to 6.187/92.881/96.233/90.673 ms. The two-pass coalesced version replaces it.
Artifacts are in `shapes-compact-norm`, `shapes-shared-norm`, and
`shared-norm-comparison.json` under the ignored profile directory.

### Reuse gradients after their final consumer

The execution plan now assigns gradient buffers by backward lifetimes instead
of allocating one for every graph node. Allocation of an operation's input
gradients happens while its output gradient remains live; only then may that
output slot be reused. Residual and XSA contributions keep their slots until
all uses are processed. Checked `get_disjoint_mut` access handles residual
copies without depending on the relative order of slot indices.

| Scenario | Shared-norm memory (MiB) | Reused gradients (MiB) | Before (ms/update) | After |
| --- | ---: | ---: | ---: | ---: |
| Original | 238 | 178 | 5.380 | 5.396 |
| Large batch | 2714 | 1738 | 80.466 | 80.637 |
| Wide model | 2608 | 1510 | 83.095 | 83.132 |
| Deep model | 2828 | 1554 | 81.686 | 81.758 |
| Combined | 2838 | 1588 | 76.854 | 76.874 |
| Long context | 2738 | 2250 | 377.517 | 378.322 |

This saves another 488--1274 MiB in the large scenarios without changing
arithmetic or kernel definitions. All six 15-update checkpoints remain
byte-identical. A new CPU property check verifies exclusive ownership of every
simultaneously live gradient, including residual branches, 64 blocks, and
mesh depths 256, and checks that allocation matches peak liveness. All 29 CPU
tests, strict Clippy, full CUDA verification, and the complete mod-97 epoch
pass. Fixed-shape measurements and comparisons are retained in
`shapes-gradient-reuse` and `gradient-reuse-comparison.json`.

### Fit tensor capacities to the workload

The benchmark selects smaller const-generic capacities for vocabulary,
short contexts, small widths, and head counts. Its fixed 99-symbol dataset fits
vocabulary capacity 128; context-four workloads no longer reserve 128 positions
per attention row. Long-context workloads with four heads no longer reserve
eight. The grokking CLI uses context capacity four while retaining support for
up to 256 symbols and eight heads. These are existing model specializations,
with the same kernels, active dimensions, and checkpoint format.

| Scenario | Reused-gradient memory (MiB) | Fitted capacities (MiB) |
| --- | ---: | ---: |
| Original | 178 | 152 |
| Large batch | 1738 | 1328 |
| Wide model | 1510 | 1452 |
| Deep model | 1554 | 1386 |
| Combined | 1588 | 1486 |
| Long context | 2250 | 1466 |

All six 15-update checkpoints and the complete mod-97 epoch checkpoint remain
byte-identical. Large-scenario update times stay within 0.4% of the preceding
stage. The original scenario's first short-warmup run measured 5.853 ms; longer
paired runs (ten warmup, 50 timed updates, three trials, repeated twice) measured
5.386/5.384 before and 5.381/5.386 after. Width eight additionally moves from
2.639 to 1.909 ms/update while owned buffers fall to 21.4 MiB. Batch one remains
0.736 ms versus 0.729; its extra normalization launch is still a useful target.
Measurements and exact comparisons are retained in `shapes-compact-capacities`,
`capacities-paired-*`, and `compact-capacities-comparison.json`.

### Coalesce token-vector lookup

Embedding lookup now gives each output channel a scalar owner. Neighboring
threads load neighboring embedding channels and write neighboring feature
channels, replacing a thread's serial whole-row copy. The owning cast restores
the original row buffer after launch; no workspace allocation or arithmetic is
added, and `_vocab` keeps its contract-only name.

| Scenario | Before (ms/update) | Coalesced lookup |
| --- | ---: | ---: |
| Original | 5.389 | 5.255 |
| Large batch | 80.316 | 73.052 |
| Wide model | 83.198 | 80.426 |
| Deep model | 81.665 | 81.473 |
| Combined | 76.883 | 75.547 |
| Long context | 376.533 | 374.781 |

Both stages use ten warmup updates and three trials. The original row uses
50 timed updates to avoid the observed short-run clock warmup; the large rows
use ten. Large-batch update time decreases by 9.0%. Its Nsight Systems trace
measures lookup at 0.146 GPU ms/update versus 8.292 in the initial expanded
trace. All six 20-update checkpoints are byte-identical, and full f64 CUDA
verification and strict Clippy pass. Width eight measures 1.882 ms/update and
batch one 0.720. Memory stays unchanged. Artifacts are in
`shapes-coalesced-lookup*`, `lookup-paired-*`, and
`coalesced-lookup-comparison.json`.

### Use freed capacity for larger stress workloads

The first enlarged workload set used the `stress-*` scenarios below, keeping the original
fixed scenarios available through `--scenario large-batch`, `--scenario wide`,
and the other original names. Timings across differently sized scenarios are
not before/after speed comparisons. The new shapes demonstrate usable capacity
freed by cache, gradient-lifetime, and tensor-capacity changes.

| Scenario | Width / blocks / batch / context | Process memory (MiB) | ms/update |
| --- | --- | ---: | ---: |
| Original | 128 / 2 / 512 / 4 | 152 | 5.311 |
| Large batch | 128 / 2 / 19200 / 4 | 3056 | 169.681 |
| Wide model | 1024 / 2 / 2560 / 4 | 3128 | 177.530 |
| Deep model | 256 / 46 / 512 / 4 | 3058 | 185.449 |
| Combined | 512 / 4 / 2688 / 4 | 3170 | 166.188 |
| Long context | 128 / 2 / 3 / 4096 | 2846 | 1141.452 |

Measurements use ten warmup and ten timed updates in three trials, with
separate process-memory probes. Batch sizes increase by 2.34x, 2.22x, and
2.24x in the large-batch, wide, and combined cases; the deep case grows from
20 to 46 blocks, and the long context grows from 1024 to 4096 tokens. These
shapes retain all FP32 state and complete sequences, with the same mesh depths.

To keep a pure large-batch scenario near 3 GiB after memory reductions, the
validated row limit increases from 65536 to 131072. CPU boundary tests follow
the new limit. Independent f64 mesh reductions cover 65535/65536/65537 and
131071/131072 rows, and tied-embedding sums cover 131072 rows. Chunk-count
contracts reject counts past 4096. Full CUDA verification, all 29 CPU tests,
and strict Clippy pass. Artifacts are in `stress-memory-baseline`.

### Tile attention score gradients and reuse the raw attention output

Score backward uses `dot(dY, Y)` for the softmax gradient's weighted mean,
where `Y` is the retained raw attention output before XSA. This follows
`sum(P * dot(dY, V)) = dot(dY, sum(P * V))` and removes the serial weighted
reduction and temporary per-score dot values. Independent warps own checked
arrays of consecutive causal scores. Owning buffer casts preserve allocation
and exclusive access; unsupported tile divisibility falls back to smaller
tiles. There are no client-side unsafe calls or new architecture flags.

The mean is computed inside each tile rather than in a separate statistics
kernel. The inline variant removes that launch and its unused RMS calculation.
With 64-score tiles for divisible capacities and smaller checked fallbacks,
median full-update times are:

| Scenario | Previous score backward (ms) | Inline tiled backward (ms) | Process memory (MiB) |
| --- | ---: | ---: | ---: |
| Original | 5.311 | 5.312 | 152 |
| Context 1024, batch 16 | 374.781 | 332.124 | 1466 |
| Context 4096, batch 3 | 1141.452 | 1012.213 | 2846 |

All measurements use ten warmup and ten timed updates in three trials.
These are fixed-shape comparisons, with unchanged allocations. A cached mean
plus one score per warp was slower: 439.920 ms at context 1024 and 1317.287 ms
at context 4096. Cached-mean 32-score tiles took 337.922 and 1034.799 ms;
inline means simplify the implementation and improve both measurements.
Inline tiles of 32 scores took 332.128 / 1019.943 ms; 128-score tiles took
333.358 / 1016.591 ms and offered no improvement over 64.

Using the algebraic identity changes FP32 summation order. After 20 updates,
the relative L2 weight difference from the previous implementation is at most
`1.59e-5` across all fixed and stress shapes, and below `1.2e-8` in both long
contexts. Inline and cached means produce identical checkpoints on those
contexts. Existing f64 gradient and optimizer tolerances are retained.
The block-count helper divides the warp count before converting to `u32`;
CPU checks cover representable block counts beyond `u32::MAX` threads and
reject block-count overflow. Full f64 CUDA verification, strict AdamW checks,
and context-dependent learning pass with unchanged tolerances.

The context-4096 Nsight Systems capture reduces score backward from 312.236
to 181.596 GPU ms/update. Forward attention remains 589.683 ms and input
backward 207.834 ms. Nsight Compute reports 89.86 ms for one score kernel,
74.12% SM and 19.71% DRAM throughput, and 48 registers/thread. In the deep
stress workload, mesh-gradient partials take 45.41 us at 80.73% DRAM throughput
and 26 registers/thread, indicating that avoiding memory traffic is a useful
next target.

Artifacts are in `shapes-score-tiles`, `stress-score-tiles`, `score-inline-32`,
`score-inline-64`, `score-inline-128`, `score-pointwise`,
`score-tiles-comparison.json`, `score-inline-long-trace.nsys-rep`,
`score-inline-long-counters.ncu-repz`, and `stress-mesh-partials-counters.ncu-repz`.

### Fuse linear mesh projections with residual addition

Each decoder block's final attention and FFN projections write the residual
sum directly. Their backward slope is one, so the removed standalone linear
output is unnecessary for derivatives. Removing two graph nodes per block
saves their retained feature buffers. Once a linear projection has finished
backward, its gradient buffer transfers to the skip input through checked
disjoint borrows and ownership swaps. This also removes the copy kernel and
standalone add operations from the graph. The Rust change removes eight lines
overall while retaining ordinary functions and the same parameters.

| Fixed scenario | Before ms/update | After ms/update | Before MiB | After MiB |
| --- | ---: | ---: | ---: | ---: |
| Original | 5.259 | 5.087 | 152 | 148 |
| Large batch | 73.110 | 70.840 | 1328 | 1264 |
| Wide | 81.087 | 78.468 | 1452 | 1380 |
| Deep | 81.836 | 78.795 | 1386 | 1306 |
| Combined | 76.079 | 73.354 | 1486 | 1406 |
| Context 1024 | 332.124 | 333.846 | 1466 | 1434 |

| Expanded stress scenario | Before ms/update | After ms/update | Before MiB | After MiB |
| --- | ---: | ---: | ---: | ---: |
| Large batch | 171.062 | 165.949 | 3056 | 2904 |
| Wide | 179.659 | 173.893 | 3128 | 2968 |
| Deep | 186.298 | 179.668 | 3058 | 2874 |
| Combined | 167.514 | 161.706 | 3170 | 2994 |
| Context 4096 | 1012.213 | 1020.830 | 2846 | 2822 |

Both sets use ten warmup and ten timed updates in three trials. Short-context
updates improve by about 3–4%. Long-context updates are 0.5–0.9% slower in
these runs while saving 24–32 MiB; the memory reduction is independently useful.
All eleven 20-update checkpoints and the complete modular-addition epoch are
byte-identical to the preceding implementation. All 30 CPU tests, strict
Clippy, f64 CUDA derivatives, strict AdamW validation, and context-dependent
learning pass. The unfused large decoder mesh itself reaches 91.31% DRAM
throughput in Nsight Compute, reinforcing the value of reducing full-buffer
passes. Artifacts are in `fused-residual`, `fused-residual-comparison.json`,
`verify-fused-residual.txt`, and `stress-decoder-mesh-counters.ncu-repz`.

### Attention recomputation experiment

Recomputing score derivatives inside aligned-warp Q/K/V backward removes the
score matrix entirely. The experimental kernel passes f64 derivatives,
strict AdamW checks, and learning, including long sequences with head widths
32/64 and padded channel capacities. It needs a small cached `dot(dY, Y)`
per head and recomputes `dot(dY, V)` in both Q and K accumulation loops.

| Scenario | Retained scores ms/update | Recomputed ms/update | Retained MiB | Recomputed MiB |
| --- | ---: | ---: | ---: | ---: |
| Context 1024, batch 16 | 333.846 | 445.346 | 1434 | 1178 |
| Context 4096, batch 3 | 1020.830 | 1356.446 | 2822 | 2054 |

The 256/768 MiB memory reduction is substantial, but update time increases
about 33%. This implementation is retained as an experiment while compact
causal score storage is investigated as a way to avoid redundant dot products.
It does not replace the default kernels. Sources, binaries, f64 verification,
and measurements remain in `gpu-recomputed-scores.rs`,
`mesh-gpt-bench-recomputed-scores`, `verify-recomputed-scores.txt`, and
`recomputed-scores` under the ignored profiling directory.

### Store only causal score tiles

The retained score-gradient buffer is now a flat FP32 slice. A query at
position `p` reserves `ceil((p + 1) / tile)` tiles rather than a full context
row. Complete bands contain `tile` rows; a final partial band reserves only
its actual rows. The producer maps each owned warp tile to its causal query
and key interval. Both CPU allocation and CUDA indexing use the same layout
functions. Integer corrections make the floating square-root estimate exact
at band boundaries, and launch contracts validate bands, tails, sequence
counts, score lengths, and outputs with checked arithmetic.

Both backward kernels use the same compile-time tile through a shared launch
helper. Q reads consecutive score cells. K advances by each future row's
compile-time rounded length, without an inner-loop band-transition branch.
Removing that branch is material: an earlier packed layout took 1095.158 ms
at context 4096, compared with 1027.658 ms for the simpler loop. Interleaving
heads within each tile was also tested and was slower on long contexts
(346.545 / 1067.281 ms at contexts 1024 / 4096), so head-major storage remains.

| Scenario | Full scores ms/update | Compact scores ms/update | Full MiB | Compact MiB |
| --- | ---: | ---: | ---: | ---: |
| Context 1024, batch 16 | 333.846 | 339.736 | 1434 | 1314 |
| Context 4096, batch 3 | 1020.830 | 1027.658 | 2822 | 2444 |

The score allocation itself shrinks from 256 to 136 MiB and from 768 to
390 MiB. Timing increases 1.8% / 0.7% in these trials while saving 120 / 378
MiB of actual process memory. The earlier loop used 40 registers/thread at
99.4% achieved occupancy, but Nsight Compute found long-scoreboard stalls
and only 5.88% DRAM throughput. Removing inner-loop control flow improved
timing despite the already high occupancy. A 50-update paired baseline
comparison measured 5.149 vs 5.100 ms; the initial short 5.383 ms measurement
was not reproduced with the longer run.

CPU coverage checks every causal tile through context 4096 at tile sizes
1/4/32/64, including band tails and inverse coordinates. Full f64 verification
adds sequences 33/129 with head widths 32/64 and padded feature capacities.
All 31 CPU tests and strict Clippy pass. Nineteen saved checkpoints across
the layout experiments are byte-identical to the preceding full-score
implementation. No new kernels, unsafe client code, flags, or macros are
required. The additional layout and contract code replaces dense index math;
attention backward launch code is shared through ordinary const generics.
Artifacts are in `packed-scores`, `packed-const`, `packed-no-branch`,
`packed-interleaved`, `packed-paired-*`, `packed-scores-comparison.json`,
and `packed-backward-counters.ncu-repz`.

### Fill the saved capacity and simplify the graph

The current runner defaults increase the workloads again after residual
fusion and causal score packing. Earlier tables remain fixed-size comparisons;
the following table demonstrates added capacity rather than speedup:

| Scenario | Width / blocks / batch / context | Process MiB | ms/update |
| --- | --- | ---: | ---: |
| Original | 128 / 2 / 512 / 4 | 148 | 5.145 |
| Large batch | 128 / 2 / 20000 / 4 | 3052 | 172.624 |
| Wide | 1024 / 2 / 2688 / 4 | 3112 | 181.799 |
| Deep | 256 / 49 / 512 / 4 | 3054 | 191.340 |
| Combined | 512 / 4 / 2816 / 4 | 2996 | 169.271 |
| Long context | 128 / 2 / 4 / 4096 | 3234 | 1372.437 |

Measurements use ten warmup and ten timed updates in three trials, and
independent memory probes. The previous stress set used batch 19200 / 2560 /
2688, 46 blocks, and batch 3 at context 4096. Score packing permits a third
more complete long sequences within roughly 3 GiB while retaining FP32 state.
Artifacts are in `stress-packed-capacity`.

The plan no longer stores each output ID in its operation: the ID is already
the operation's index. One ordinary `push` function handles node insertion,
and input lists still drive checked gradient liveness. Mesh forward and input
backward share a const-generic kernel definition with distinct compiled
specializations. The combined simplification removes 46 Rust lines without
new macros or allocations. Full f64 verification, all 31 CPU tests, strict
Clippy, and the complete epoch pass. Six fixed-workload checkpoints and the
epoch are byte-identical. Original / large / wide / deep / combined / long
medians are 5.114 / 71.167 / 78.555 / 79.180 / 73.538 / 341.522 ms,
with the same memory as the preceding implementation. Artifacts are in
`shared-mesh`, `shared-mesh-comparison.json`, and `verify-shared-mesh.txt`.

The mesh entry prefix is now `gpt_mesh_TID_` for both compiled specializations.
To select the first input-backward launch with Nsight Compute, skip the matching
forward calls: `attention_depth + blocks * (4 * attention_depth + ffn_depth)`.
This is 48 calls for the original benchmark. `--launch-skip 4` still selects
its first decoder forward projection.
