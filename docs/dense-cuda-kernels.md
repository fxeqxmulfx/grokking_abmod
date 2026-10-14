# Tiled CUDA Dense products

Large Dense layers (at least 512 input or output channels) use a shared-memory
16×16 matrix product for forward propagation and both gradients. Small layers
retain the existing CUDA kernels. This accelerates the native Rust optimizer
search; the website continues to train locally on CPU/WASM.

Inputs and weights are packed into padded column panels. The product keeps
the original ascending reduction order, then unpacks ordinary row-major
outputs. Workspaces are cached by matrix shape. The existing safe CUDA Rust
`block_pipeline` API stages read-only tiles and supplies block synchronization;
checked launch contracts enforce tensor sizes and padding. No client `unsafe`
or dependency change was needed.

## Verification

The locked release build used `cargo oxide build --arch sm_86`, CUDA Rust
revision `5276d28f89b56342f124b7fad8cf081f05e2a6d5` from `Cargo.lock`.
`mesh-gpt grok-simple --verify-model tied-rotary-residual-dense-mlp` passed.

Fifteen additional matrix cases exercise identity, ReLU and square activations,
single rows, odd dimensions, offsets, padded input strides, 48-row tails,
512-row minibatches, and 1024-channel matrices. Every forward output, input
gradient and weight gradient is compared with an independent f64 sum, allowing
at most `1e-5 × max(1, largest reference magnitude)`. Existing end-to-end model,
AdamW, repeated-run and invalid-input checks also pass.

For W1024, seed 11, normal-width embeddings, ten complete training epochs with
full-split evaluation took 28.647765 seconds with the original CUDA binary and
3.320558 seconds with the tiled binary (8.63× in this local pilot). Final f32
weights matched bit for bit, SHA-256
`6af3f235dd462d89b4bf5920c7ee077433e510c357f04855e31e9fa073879bb4`.
The original and tiled control also reached 99% held-out accuracy at epoch 270.
This is a measured pilot, not a general hardware throughput guarantee.

Evidence: `docs/grokking-results/dense-cuda-verification/`.

## Optimizer search

`experiments/div97-dense-optimizer-grid.json` defines the finite grid and
independent-seed confirmation policy. With Node 24.21.0, run
`node scripts/optimizer-grid.mjs`. The runner saves each native CUDA trial's
arguments, log and learning curve, resumes completed trials, and aborts on a
failed trial. Small widths use the existing verified CUDA binary; W1024 waits
for `target/optimizer-grid/cuda-ready.json`, written only after correctness and
checkpoint verification of the tiled binary. All training executes in Rust
on CUDA. The optimization objective is the earliest 99% held-out crossing,
without requiring a memorization plateau.
