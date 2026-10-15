> Historical uniform-E baseline. The current TypeScript frontend uses the
> [Normal(0, 1/W) embedding variant](rotary-dense-embedding-results.md).

# Rotate90 Dense + ReLU: Rust results

The dense replacement keeps the shared operand/readout embedding and changes
the single residual branch to a full width-by-width matrix followed by ReLU:

```text
E: 97 x W, A: W x W
x0 = E[a] + J(E[b])       # J(u, v) = (-v, u) on coordinate pairs
r = RMSNorm(x0)
x1 = x0 + ReLU(r A)
h = RMSNorm(x1)
logits = h E^T
```

`A` uses seeded Kaiming normal initialization with variance `2/W` (fan-in,
ReLU gain). `E` retains the seeded uniform initialization. There are no biases
or learned RMSNorm parameters. This is a dense model; it does not satisfy the
two-parent constraint proposed for the separate full-mesh experiment.

Both runs use division modulo 97, a 50/50 split of the 9,312 equations, seed 11,
one residual block, minibatches of 512 with a 48-row tail, and AdamW with learning
rate 0.001, weight decay 0.1, betas 0.9/0.98, epsilon 1e-8 and a 10-update warmup.
Both complete splits are evaluated every epoch. The budget is 1,000 epochs;
training stops at held-out accuracy of at least 99%.

| Width | Parameters | First train >=99% | Train >=99%, test <=10% plateau | Last test <=10% | First test >=90% | Rise | First test >=99% |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 28,800 | 339 | None | 331 | 588 | 257 epochs | 679 |
| 1024 | 1,147,904 | 15 | Epochs 15–179: 164 epochs | 179 | 320 | 141 epochs | 368 |

Final held-out accuracy is 99.055% for W128 and 99.119% for W1024. W1024 passes
the existing criterion: at least 25 elapsed epochs with train >=99% and test
<=10%, a <=250-epoch transition from the last test <=10% observation to the
first test >=90%, and a later test >=99%. W128 generalizes without a qualifying
plateau. These are single-seed observations, not evidence of seed robustness.

The W128 run took 18.68 seconds; W1024 took 1,042.57 seconds on the local RTX
3050 Laptop GPU. Complete evaluations and host-side reports are included.

[Archived metrics and provenance](grokking-results/rotary-dense-relu/summary.json)
include checkpoint/history hashes and the unchanged analysis policy. The
[learning curves](grokking-results/rotary-dense-relu/short-mesh-seeds.svg) show
both complete runs; CSV histories are under the archive's `raw/` directory.
The replay arguments are in
[`experiments/div97-rotary-dense-one-block.args`](../experiments/div97-rotary-dense-one-block.args).

Validation before training: 75 Rust tests, strict Clippy, formatting, the locked
`sm_86` CUDA release build, and the complete simple-model CUDA verification
suite passed. The new dense paths were checked against independent f64
automatic differentiation for logits, losses, all parameter gradients,
gathered/offset minibatches, AdamW and repeatability.

## Local browser implementation

Both local tabs now implement this Dense + ReLU architecture, retaining
pre-RMSNorm inside the residual branch and final RMSNorm before the shared Eᵀ.
The [frontend documentation](../web/own-channel/README.md) records the
handwritten WASM/scalar/SIMD and WebGPU implementations, all-weight visualization,
localized SVG/type/equation pages and complete-split browser measurements.
Seed-11 initial checkpoints match Rust byte-for-byte at both widths. All 105
Node tests and both WebGPU f64 gradient/optimizer suites passed.

| Width / browser engine | First test >=90% | First test >=99% | Memorization plateau | Rise | Qualifies |
| --- | ---: | ---: | --- | ---: | --- |
| 128 / WASM SIMD | 577 | 657 | None | 245 epochs | No |
| 128 / WebGPU | 562 | 645 | None | 238 epochs | No |
| 1024 / WebGPU | 516 | 634 | 15–192: 177 epochs | 324 epochs | No |

The browser W1024 run reaches 99% but its transition exceeds the unchanged
250-epoch limit. The UI correctly reports generalization rather than grokking.
The native W1024 qualification therefore does not establish browser qualification
or multi-seed robustness. Complete WASM W1024 convergence has not been measured;
its numerical kernels and real UI training were checked.

Browser histories, final weights, criteria and source hashes are archived under
[`web/own-channel/tests/results/dense-rotary*-seed11/`](../web/own-channel/tests/results/).
Training remains frontend-only on localhost; no cloud update was published.

## Shared E initialization control

A subsequent [matched Rust study](rotary-dense-embedding-results.md) initializes
E with Gaussian variance 1/W while preserving the exact initial A, split and
optimizer. On seed 11, first test >=99% moves from 679 to 605 epochs at W128
and from 368 to 270 at W1024. The latter retains a qualifying 92-epoch plateau
and a 116-epoch test rise. These are single-seed observations. The initializer
is opt-in through `--dense-embedding-initialization normal-width`; the local
browser and Rust default retain the original uniform E.
