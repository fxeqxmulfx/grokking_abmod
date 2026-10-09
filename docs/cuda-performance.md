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

RMSNorm and QK-Norm now assign one thread per output channel when the capacity
and active head width are multiples of 32. Each complete warp cooperatively
reduces squared inputs or the backward dot product. Smaller and unaligned
shapes use the original serial reduction. The cache stores each channel's
normalized value beside its inverse RMS so typed disjoint writes remain safe.
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

For more than 128 token rows and widths of at least 32, tied-embedding gradients
use independent chunks of 128 rows. Each partial combines the repeated input
lookup and dense output-head contributions in FP32. The second kernel reduces
directly into feature-row storage, avoiding a separate serial conversion. Small
and narrow workloads retain the direct kernel. The chunk length is a Rust const
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
  --kernel-name 'regex:gpt_embedding_backward_TID_.*' \
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
