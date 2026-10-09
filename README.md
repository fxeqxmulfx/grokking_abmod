# grokking_abmod

A Mesh Neural Net in Rust using [CUDA Oxide from the `experiments` branch](https://github.com/fxeqxmulfx/cuda-rust/tree/experiments).
The example learns the piecewise function from the **Mesh Neural Nets** section of
Stephen Wolfram's [What's Really Going On in Machine Learning? Some Minimal Models](https://writings.stephenwolfram.com/2024/08/whats-really-going-on-in-machine-learning-some-minimal-models/).
Forward propagation, backpropagation, gradient reduction, and Adam updates run on the GPU.
`cuda-core`, `cuda-device`, and `cuda-host` are pinned to the same commit in `Cargo.lock`.

## Model

Each neuron receives at most two neighboring inputs, with independent edge
weights and a bias. Every layer, including the output, uses ELU with alpha 1.
The target is 1 on `[-2, -1)` and `[0, 2)`, and 0 elsewhere.

The article builds an expanding, zero-padded mesh and selects its central
output. Removing neurons that cannot reach that output gives the equivalent
diamond used here. At the default depth of 14, the layer widths, including
the input, are:

```text
1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 7 -> 6 -> 5 -> 4 -> 3 -> 2 -> 1
```

This leaves 63 neurons, 112 trainable edge weights, and 63 biases: **175
parameters**. Edges from zero padding need no parameters. CPU tests verify
equivalence to the full expanding mesh. Even depths from 2 to 32 are supported.
For half-depth `h`, the counts are `h(h + 2)` neurons and `h(3h + 4)` parameters.

Training uses mean squared error and full-batch Adam (`beta1 = 0.9`,
`beta2 = 0.999`, `epsilon = 1e-8`). Defaults are 1024 uniformly sampled inputs
in `[-3, 3)`, seed 7, 3000 steps, and learning rate 0.01. Initialization uses
independent weights scaled by fan-in and small random biases. These training
settings are a reproducible example, rather than the article's exact experiment.

## Implementation

- `src/mesh.rs`: validated topology, initialization, target function, and an
  independent f64 CPU reference.
- `src/gpu.rs`: CUDA kernels and the safe `GpuMesh` API for prediction, parameter
  replacement, gradient inspection, and training on caller-provided scalar data.
- `src/main.rs`: training example, CSV export, and GPU correctness checks.
- `scripts/plot-mesh.py`: prediction, loss, and learned-weight mesh plots.

The training batch, model parameters, per-sample gradients, and optimizer state
stay on the GPU throughout the update loop. One thread owns each sample during
forward/backward propagation; one thread owns each parameter during gradient
reduction and Adam. Typed launch contracts and `DisjointSlice` enforce exclusive
writes without atomics. `GpuMesh` keeps descriptors private and validates shapes,
finite inputs, and options. Its constructor contains one documented `unsafe`
call to bind the crate's own embedded CUDA artifact; public client calls are safe.
Training batches are limited to 65536 samples; gradient storage is approximately
3.25 KiB per sample (the fixed capacity for depth 32).

Successive `train` calls continue Adam's moments and step history.
`set_parameters` replaces weights and resets the optimizer. `loss_and_gradient`
computes each sample's gradient on the GPU and averages it on the host for
diagnostics; the training path reduces gradients on the GPU.

## Setup

Requires Linux, an NVIDIA GPU with a working driver, CUDA Toolkit 13.4,
LLVM/Clang 23, Git, and Python 3.
Rust `nightly-2026-10-09` and its required components are specified in `rust-toolchain.toml`.
LLVM paths are configured in `.cargo/config.toml`. For a different installation,
adjust these paths or override them with `CC`, `CXX`, `LIBCLANG_PATH`, and `CUDA_OXIDE_LLC`.

From the project root:

```bash
./scripts/setup-cuda-oxide.sh
cargo oxide doctor
cargo oxide run
```

The script builds the backend and `cargo-oxide` from the dependency commit in `Cargo.lock`.
Sources and the backend are stored in `target/`. `cargo-oxide` is installed in
Cargo's binary directory (usually `~/.cargo/bin`).

## Train and inspect

```bash
# Train the default 14-layer mesh and export artifacts.
cargo oxide run

# Change model/training settings; application options follow --.
cargo oxide run -- --depth 14 --samples 4096 --epochs 10000 --seed 7

# Inspect all options.
cargo oxide run -- --help

# Optional plots (requires NumPy and Matplotlib).
python3 scripts/plot-mesh.py
```

The default output directory is `target/mesh-net`; change it with `--output PATH`.
It contains `loss.csv`, `predictions.csv`, `parameters.csv`, `nodes.csv`, and
`run.txt`. The plot script adds `mesh-training.svg` and `mesh-training.png`.
Predictions are evaluated on a separate 2001-point grid over `[-3, 3]`.

A run on an RTX 3050 Laptop GPU produced:

```text
ELU diamond: depth=14, neurons=63, weights=112, biases=63, parameters=175
epoch      0: training MSE = 0.63313175
epoch   3000: training MSE = 0.00296509
Evaluation MSE: 0.61336397 -> 0.00343892
```

Results can vary slightly with floating-point arithmetic across GPUs. The network
approximates the discontinuities with continuous transitions.

## Build and verify

```bash
# Build for RTX 3050 Laptop / Ampere without running on the GPU.
cargo oxide build --arch sm_86 -- --release --locked

# Check topology, pruning equivalence, and every CPU gradient with finite differences.
cargo test --lib --locked

# Compare GPU outputs, MSE, gradients, and Adam against the CPU reference.
cargo oxide run -- --verify-only

# Check formatting and lint all targets.
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
```

GPU verification covers depths 2, 6, 14, and 32, batch sizes 1, 7, and 129
(including partial CUDA blocks), optimizer continuation/reset, parameter round-trips,
and rejected invalid input. On success, it prints:

```text
CUDA Mesh Neural Net verification passed
```

Rebuild the tools after updating dependencies:

```bash
cargo update -p cuda-core -p cuda-device -p cuda-host
./scripts/setup-cuda-oxide.sh
```

Run setup again after deleting `target/` as well.

The local article archive in `papers/` is excluded from Git by `.gitignore`.
