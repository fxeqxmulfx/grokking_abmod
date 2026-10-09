# grokking_abmod

Rust CUDA kernels using [CUDA Oxide from the `experiments` branch](https://github.com/fxeqxmulfx/cuda-rust/tree/experiments).
`src/main.rs` contains a GPU vector addition example and verifies all 1024 results.
`cuda-core`, `cuda-device`, and `cuda-host` are pinned to the same commit in `Cargo.lock`.

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

## Build and verify

```bash
# Build for RTX 3050 Laptop / Ampere without running on the GPU.
cargo oxide build --arch sm_86 -- --release --locked

# Build and run with automatic GPU detection.
cargo oxide run

# Check formatting.
cargo fmt --check
```

On success, the program prints:

```text
CUDA vecadd: all 1024 elements correct
```

Rebuild the tools after updating dependencies:

```bash
cargo update -p cuda-core -p cuda-device -p cuda-host
./scripts/setup-cuda-oxide.sh
```

Run setup again after deleting `target/` as well.
