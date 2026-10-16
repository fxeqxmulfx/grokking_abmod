# CUDA Dense products

Every Dense layer, the tied readout, and the rotary embedding with its
gradient use register-blocked CUDA matrix products (`src/simple/dense_tiled.rs`).
Per-row work runs in small kernels in `src/simple/gpu.rs`. This accelerates the
native Rust optimizer search; the website continues to train locally on
CPU/WASM. The safe CUDA Rust `block_pipeline` API stages read-only tiles and
supplies block synchronization; checked launch contracts enforce tensor sizes
and padding. No client `unsafe` or dependency change was needed.

## Matrix products

Operands are packed into K-major panels 16 times the register block wide, one
thread per panel cell. Each `blocked_product` thread block has 256 threads:
thread `16 y + x` keeps a `TM × TN` block of adjacent outputs in registers and
reads its operands from shared memory with vector loads. The reduction advances
16 rows per stage, and each stage's loop is unrolled eight times, which keeps
the 8×8 block in registers. `unpack` writes ordinary row-major outputs and
applies the activation slope to input gradients. Workspaces are cached by
length.

The register block is the largest of 8×8, 8×4, 4×4, 2×2 and 1×1 that still
gives every multiprocessor a thread block. A block is skipped when the next
smaller one gives as many thread blocks, since it would only pad more. Products
too small for any of them use 1×1. The rotary gradient reads three values per
operand cell, so it chooses among 4×4, 2×2 and 1×1 by the same rule.

## Exact results

Every output accumulates from +0 in ascending reduction order with fused
multiply-adds, the order of a sequential loop. Register blocks, unrolling and
thread-block counts therefore change speed but never a bit of the result.
Reductions are never split. Reduction padding multiplies −0 by +0, and adding
−0 leaves every partial sum unchanged, including an underflowed −0.

The per-row kernels keep the orders of the kernels they replace:

- The cross-entropy gradient first computes each row's maximum and its sum of
  `exp(x − max)` in class order, one warp per row, instead of recomputing the
  softmax for every logit.
- Evaluation finds each row's first class with the highest logit on the device,
  one warp per row, and marks rows that hold a value that is not finite.
- RMSNorm computes and applies each row's statistics in one pass, one warp per
  row, at power-of-two widths from 16 to 1024. Other widths keep the two-kernel
  path.

The compiler fuses a multiply into a following addition wherever it sees
both, so a kernel matches only if it exposes the same pairs. One-pass RMSNorm
therefore bounds its reduction loop by the run-time width, as the statistics
kernel does. With a compile-time bound the compiler unrolled the loop: at width
32 it fused each lane's only square into the first shuffle sum, and at widths
64 to 512 it fused the first square into the second rather than the second
into the first. Either change alters the inverse RMS of only 0.4% to 6% of
rows, so verification compares one-pass and two-pass RMSNorm on 4096 rows per
width.

## Host side

Kernel launches are prepared once per specialization and configuration.
Evaluation queues both splits before the first copy back and copies logits
into reused page-locked buffers. A writer thread scores rows and writes each
record while training continues. It scores rows in parallel chunks and keeps
their order, so every sum is unchanged.

## Verification

The locked release build used `cargo oxide build --arch sm_86`, CUDA Rust
revision `5276d28f89b56342f124b7fad8cf081f05e2a6d5` from `Cargo.lock`.
`mesh-gpt grok-simple --verify-model tied-rotary-residual-dense-mlp` passed:

- Eighteen Dense cases cover identity, ReLU and square activations, single
  rows, odd dimensions, offsets, padded input strides, 48-row tails, 512-row
  minibatches and 1024-channel matrices. Twelve embedding cases cover the tied
  readout and its input gradient, plus, with identity, the rotary embedding and
  its gradient. Every output and gradient is compared with an independent f64
  sum, allowing at most `1e-5 × max(1, largest reference magnitude)`. Each
  register block is then forced in turn and must match the automatic choice
  bit for bit.
- The warp-per-row predictions and softmax maxima match one-thread scans bit
  for bit, and softmax denominators match f64, on rows with ties, signed zeros
  and values that are not finite. One-pass RMSNorm matches the two kernels bit
  for bit, forward and backward.
- The existing model, AdamW, repeated-run and invalid-input checks pass, and so
  do the 66 library tests.

Nine end-to-end runs were compared against the committed binary, which uses
16×16 shared-memory tiles for Dense layers with at least 512 channels and the
original kernels otherwise. Each used normal-width embeddings, ReLU, one-hot
inputs and batch 512 on div 97. They covered widths 16 to 1024, learning rates
0.001 to 1, weight decays 0 to 1, seeds 11, 17 and 23, and two runs that stop
at 99% held-out accuracy. The learning curves (apart from wall time), run
metadata, initial and final weights, and predictions were identical bit for
bit. The W1024 seed 11 checkpoint after ten epochs still has SHA-256
`6af3f235dd462d89b4bf5920c7ee077433e510c357f04855e31e9fa073879bb4`, as with
the original CUDA kernels. Seven trials archived in
`docs/grokking-results/dense-optimizer-grid/`, at widths 16 to 1024 and
trained by both binaries that search used, were rerun with their saved
arguments; their learning curves and run metadata matched apart from wall
time.

Milliseconds per epoch (ten optimizer steps and the evaluation of both splits)
on an RTX 3050 Laptop GPU with 16 multiprocessors, at learning rate 0.001,
weight decay 0.1 and seed 11. Each value is the median of three runs of 300
epochs (60 at W1024) that alternated with the other binary:

| Width | Committed kernels | Register blocks | Speedup |
| ---: | ---: | ---: | ---: |
| 16 | 10.69 | 3.11 | 3.4× |
| 32 | 12.69 | 3.25 | 3.9× |
| 128 | 28.20 | 5.09 | 5.5× |
| 1024 | 315.03 | 38.20 | 8.2× |

These are measured local runs, not a general hardware throughput guarantee.
Each process also spends about one second loading CUDA modules.

Evidence: `docs/grokking-results/dense-cuda-register-blocks/`; for the 16×16
tiles, `docs/grokking-results/dense-cuda-verification/`.

## Optimizer search

`experiments/div97-dense-optimizer-grid.json` defines the finite grid and
independent-seed confirmation policy. With Node 24.21.0, run
`node scripts/optimizer-grid.mjs`. The runner saves each native CUDA trial's
arguments, log and learning curve, resumes completed trials, and aborts on a
failed trial. `target/optimizer-grid/cuda-ready.json`, written only after the
checks above, names a verified binary and the widths it serves, W1024 alone if
it lists none. Other widths use the existing verified CUDA binary, and W1024
waits for the record. `source.json` lists the SHA-256 of every binary the
trials used. All training executes in Rust on CUDA. The optimization objective
is the earliest 99% held-out crossing, without requiring a memorization
plateau.

The completed grid ran 437 trials, 152 screening and 285 confirmation runs, and
is archived in `docs/grokking-results/dense-optimizer-grid/`.
`experiments/div97-dense-optimizer-grid-64-256-512.json` applies the same grid
and policy to W64, W256 and W512. Its 270 trials, 90 screening and 180
confirmation runs, are archived in
`docs/grokking-results/dense-optimizer-grid-64-256-512/`. The winners from W16
to W256 are the website's fast optimizer presets; the website does not offer
W512 or W1024:

| Width | Learning rate | Weight decay | First 99% held-out, seeds 11 / 17 / 23 |
| ---: | ---: | ---: | ---: |
| 16 | 0.1 | 0 | 1033 / 407 / 163 |
| 32 | 0.1 | 0.0001 | 39 / 43 / 27 |
| 64 | 0.1 | 0.1 | 21 / 33 / 29 |
| 128 | 0.01 | 1 | 30 / 32 / 30 |
| 256 | 0.01 | 1 | 32 / 33 / 27 |
| 512 | 0.01 | 1 | 24 / 25 / 25 |
| 1024 | 0.01 | 1 | 35 / 26 / 30 |

`scripts/late-generalization-search.mjs` searches one width in the opposite
direction. Among configurations that reach 99% held-out accuracy on all three
seeds within the same 10,000 epochs, it picks the one whose earliest crossing
comes latest. Trials use the grid's CUDA settings and stop at 99%. Conclusive
trials of both grids and the plateau search are reused, and a configuration is
abandoned after its first seed that misses 99%. The winners are the website's
slow presets:

| Width | Winner | First 99% held-out, seeds 11 / 17 / 23 | Trials run | Trials reused |
| ---: | --- | ---: | ---: | ---: |
| 16 | lr 0.2, wd 0.001 | 5182 / 1333 / 7117 | 48 | 11 |
| 32 | lr 0.0003, wd 0.1 | 5199 / 3885 / 5462 | 70 | 37 |
| 64 | lr 0.0001, wd 0.0001 | 6513 / 6583 / 8501 | 114 | 36 |
| 128 | lr 0.00005, wd 0.3 | 8151 / 7931 / 8899 | 123 | 42 |
| 256 | lr 0.00005, wd 0.3 | 7632 / 6777 / 6593 | 96 | 47 |

W32 to W256 searched learning rates 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1
and 1 with weight decays 0, 0.0001, 0.001, 0.01, 0.1, 0.3 and 1. Lower
learning rates stretch the learning curve (see below), so W64 to W256 also
searched 0.00003 and 0.00005. At W32, lr 0.0001 already misses 99% on seed 11
with every weight decay. Below lr 0.0001, only weight decay 0.3 or 1 reaches
99% on every seed: lr 0.00005 with wd 1 at W64, and lr 0.00005 with wd 0.3 or
1 and lr 0.00003 with wd 1 at W128 and W256. The W128 search first ran without
abandonment, weight decay 0.3 or the two lowest learning rates: 89 trials and
37 reused ones covered every seed of its 42 configurations. Extending it to 63
configurations ran 34 more and reused 5. W16 searched learning rates 0.02,
0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7 and 1 with weight decays 0,
0.0001, 0.001 and 0.01. At W16 the grid's weight decays 0.1 and 1 miss 99% on
every confirmation seed at learning rates 0.1 and 1, so the search stops at
0.01. Within it, only lr 0.1 with wd 0 and lr 0.15 and 0.2 with wd 0.001
succeed on all three seeds. Archives are in
`docs/grokking-results/dense-optimizer-late/`, one directory per width.

The summary also ranks the configurations that succeed on every seed by the
memorization plateau of their shortest seed, by the website's rule below,
counting plateaus from 25 epochs. The longest is the website's plateau preset.
Where it is also the slow winner, the plateau preset takes the longest plateau
that groks on every seed instead. No configuration up to W64 plateaus on every
seed, so W16 to W64 have no plateau preset:

| Width | Plateau preset | Plateau, seeds 11 / 17 / 23 | Rise to 90% held-out | First 99% held-out | Grokking on every seed |
| ---: | --- | ---: | ---: | ---: | --- |
| 128 | lr 0.0003, wd 0 | 166 / 150 / 210 | 1092 / 1362 / 892 | 3375 / 3770 / 2880 | No |
| 256 | lr 0.001, wd 0.3 | 76 / 69 / 58 | 129 / 126 / 115 | 368 / 346 / 336 | Yes |

At W128 only lr 0.0003 and 0.001 with weight decay up to 0.01 plateau on
every seed, and none groks: at lr 0.001 the plateau lasts 26–51 epochs, but
held-out accuracy then takes 303–443 epochs to reach 90%. At W256 the slow
winner has the longest plateau, 1,644 / 1,403 / 1,362 epochs, followed by a
2,126–2,721-epoch rise. Three W256 configurations grok on every seed: lr 0.001
with wd 0.3, lr 0.0003 with wd 1 (plateaus of 41–78 epochs) and lr 0.003 with
wd 0.1 (38–51).

## Memorization plateau

`scripts/plateau-search.mjs` measures where the website's memorization plateau
appears and how long it gets. The website's rule (`ShortGrokking` in
`web/own-channel/src/model.ts`) takes the longest run of consecutive epochs
with training accuracy at least 99% and held-out accuracy at most 10%, before
held-out accuracy first reaches 90%. It counts a run as delayed grokking when
that plateau lasts at least 25 epochs and held-out accuracy rises from its
last epoch at most 10% to 90% within 250 epochs. Trials use the grid's CUDA
settings and 10,000-epoch budget, and stop at 99% held-out accuracy. Seed 11
screens each learning rate and weight decay. Per width, three screens are
repeated on seeds 17 and 23: the longest plateau, the longest that reaches
99%, and the longest delayed one. Conclusive grid and late-search trials are
reused. Widths outside the grid run on the same verified binary: `verify-model`
checks its products and the whole model at shape-generic sizes. Saved trials
rerun beside four concurrent ones reproduce their learning curves and
checkpoints bit for bit.

W32 to W112 searched learning rates 0.0001, 0.0003, 0.001, 0.003 and 0.01
with weight decays 0, 0.01 and 0.1. W128 to W1024 also searched weight decays
0.3 and 1. W112 and W120 ran all three seeds at learning rates 0.0001, 0.0003
and 0.001 with weight decay 0, which is all W120 ran. Lower learning rates
extend three widths: W256 to 0.00005 with weight decays 0.3 and 1, W512 to
0.00003 and 0.00005 with weight decays 0.1, 0.3 and 1, and W1024 to 0.00001
with weight decay 1. Plateaus are in epochs, for seeds 11 / 17 / 23. The
held-out column is the lowest over all settings and seeds:

| Width | Held-out at first train ≥99% | Longest plateau | Longest plateau on all seeds that reach 99% | Longest delayed plateau on all seeds |
| ---: | ---: | --- | --- | --- |
| 32 | 90.9% | — | — | — |
| 64 | 71.3% | — | — | — |
| 96 | 22.4% | — | — | — |
| 112 | 9.7% | 1 | — | — |
| 120 | 6.4% | 505, lr 0.0001, wd 0, seed 11 | — | — |
| 128 | 4.5% | 752, lr 0.0001, wd 0, seed 23 | 166 / 150 / 210, lr 0.0003, wd 0 | — |
| 256 | 1.9% | 1,644, lr 0.00005, wd 0.3, seed 11 | 1,644 / 1,403 / 1,362, lr 0.00005, wd 0.3 | 76 / 69 / 58, lr 0.001, wd 0.3 |
| 512 | 1.8% | 2,540, lr 0.00003, wd 0.1, seed 11 | 2,225 / 2,201 / 2,181, lr 0.00003, wd 0.3 | 232 / 230 / 229, lr 0.0003, wd 0.3 |
| 1024 | 0.9% | 3,632, lr 0.00001, wd 1, seed 23 | 3,484 / 3,436 / 3,632, lr 0.00001, wd 1 | 357 / 351 / 369, lr 0.0001, wd 1 |

- Up to W112 no setting forms a plateau, because memorization and
  generalization overlap. When training accuracy first reaches 99%, held-out
  accuracy is already at least 22% up to W96. At W112 it is 9.7% at best and
  passes 10% one epoch later.
- W120 plateaus on some seeds: 505 / 26 / 66 epochs at lr 0.0001, where only
  seed 23 reaches 99%, and 120 / 33 / 0 at lr 0.0003. From W128, lr 0.0003
  with weight decay up to 0.01 plateaus on every seed, and every seed reaches
  99% (2,518–4,478 epochs).
- At a fixed learning rate, the plateau is longest at intermediate width. At
  lr 0.0001 and wd 0 it lasts 433–752 epochs at W128, 1,289–1,530 at W256,
  827–843 at W512 and 471–501 at W1024. Wider models memorize sooner, at about
  3,300–4,300, 950–1,000, 275–290 and 90 epochs. But held-out accuracy also
  leaves chance sooner, at about 3,800–5,000, 2,300–2,500, 1,100 and 560–590.
- Long plateaus rise slowly. At W256 with lr 0.0001 and wd 0, held-out
  accuracy peaks at 79–93% within 10,000 epochs. With wd 0.1 every seed
  reaches 99% after plateaus of 990–1,217 epochs, the longest that generalizes
  at lr 0.0001 or above. The rise also lengthens as the learning rate falls,
  so the 250-epoch limit admits only shorter plateaus. Robust delayed grokking
  first appears at W256. Its longest plateau is at W1024 with lr 0.0001 and
  wd 1: 351–369 epochs, then a 187–203-epoch rise and 99% at epochs 743–792.
- At lr 0.0001 and below, the whole learning curve stretches as 1/lr. At
  W1024 with wd 1, lowering lr from 0.0001 to 0.00001 moves memorization from
  epochs 89–90 to 806–816, the plateau's end from 441–458 to 4,252–4,438 and
  99% from 743–792 to 7,240–7,763. A plateau in epochs is therefore bounded by
  the 10,000-epoch budget. This one is the longest found, 3,436–3,632 epochs,
  and every seed reaches 99%. Measured in lr × epochs it is only 0.034–0.036,
  while W256 at lr 0.0001 reaches 0.13–0.15 with wd 0 and 0.10–0.12 with
  wd 0.1.

The archive is in `docs/grokking-results/dense-plateau/`.

## Local residual and scalar regression experiments

The rotary dense model supports `grok-simple --no-residual`: each block becomes
`ReLU(Dense(RMSNorm(h)))`. The pre-block and final RMSNorm, shared E, tied
readout, matrices and initialization remain identical. The flag is rejected
for other architectures. The independent f64 reference and CUDA verification
include the ablation with repeated blocks and gathered minibatches.

`scalar-mlp` is a separate numeric-regression baseline. It reads two float32
features, applies an affine/ReLU input layer, optional additional hidden
layers of the same width, and one affine scalar readout. Hidden matrices use
Kaiming normal variance `2/fan_in`; the linear readout uses variance
`1/fan_in`. Every affine layer has a trainable bias, initialized to zero by default. Depth
counts all ReLU hidden layers. `--pre-rmsnorm` normalizes inputs to hidden
layers after the first; `--residual` adds skips around those same layers.
Neither applies to the two-feature input or scalar readout, and there is no
final RMSNorm. Depth 1 therefore has one common baseline with both flags off.
RMSNorm has epsilon 1e-5 and no trainable gain or bias.

Forward products, bias/activation operations, MSE derivatives, all parameter
gradients and AdamW updates run on CUDA through checked launch contracts.
Tables, parameters and optimizer moments stay on the GPU. Rust prepares the
seeded epoch order and aggregates evaluation metrics; the scalar f64 tape is
used only for verification. `--scale unit` divides operands and targets by 96;
`--scale raw` leaves them as numeric residue values. The loss is mean squared
error in those training units. Reports also give RMSE in original residue
units and exact answer accuracy after rounding, without clipping or modulo
reduction. A training-mean constant predictor supplies the MSE baseline.

```sh
cargo oxide build --arch sm_86 --cargo-target-dir target/local-ablations/build -- --release --locked --bin mesh-gpt --bin scalar-mlp
target/local-ablations/build/release/mesh-gpt grok-simple --verify-model tied-rotary-residual-dense-mlp
target/local-ablations/build/release/scalar-mlp --verify
node scripts/local-rotary-ablation.mjs target/local-ablations/build/release/mesh-gpt target/local-ablations/rotary
node scripts/local-scalar-grid.mjs target/local-ablations/build/release/scalar-mlp target/local-ablations/scalar
node scripts/summarize-local-ablations.mjs target/local-ablations
uv run --locked scripts/plot-local-ablations.py target/local-ablations
```

The [local plan](../experiments/local-residual-regression.json) fixes paired
W128 comparisons for Fast, Plateau and Slow, and a staged scalar grid over
division/multiplication, raw/unit scales, width, depth, norms, skips, lr and
weight decay. Scalar screens use 1,000 epochs on seed 11. Each selected fixed
configuration runs for 10,000 epochs on seeds 11, 17 and 23. Selection uses
the held-out split, so these are exploratory validation results. The drivers
retain exact arguments, manifests, complete evaluation histories, initial and
final weights, and nonfinite statuses. Artifacts stay under ignored `target/`;
the scripts do not commit, push, deploy or upload results.

The [MLP diagnostic follow-up](../experiments/local-mlp-diagnostics.json)
retains this original grid and adds smaller moduli, seeded subsets of the
training half, and deeper networks with 8 and 16 hidden layers. It selects
deep configurations separately for training fit and held-out fit. Both
selection criteria use the same three confirmation seeds. Planned adaptive
capacity and optimizer comparisons must record their concrete settings before
launch. These controls distinguish poor training fit from poor generalization;
a failed finite grid does not prove that a ReLU MLP cannot represent the task.

Diagnostic CLI options include `--modulus`, `--train-limit`, `--scale centered`,
`--schedule cosine`, `--min-lr-ratio` and `--beta2`. Unit scaling divides by
`modulus - 1`; centered scaling also subtracts 0.5 from both operands and the
target. Predictions are restored to original residue units before rounding.
The cosine schedule retains the ten-update warmup and reaches the requested
minimum learning-rate ratio at the last update. A train limit takes a prefix
of the seeded training half; the original held-out half remains disjoint and
unchanged, and unused training equations do not become held-out examples.

```sh
node scripts/verify-scalar-diagnostics.mjs ORIGINAL_BINARY DIAGNOSTIC_BINARY target/local-ablations/diagnostic-verification
node scripts/local-mlp-diagnostics.mjs DIAGNOSTIC_BINARY target/local-ablations controls
node scripts/local-mlp-diagnostics.mjs DIAGNOSTIC_BINARY target/local-ablations deep
```

The diagnostic driver requires the original experiment to finish first. Freeze
its binary and source snapshot before rebuilding the extended CLI. The replay
check compares all default-mode metrics, both checkpoints and final saved
predictions bit for bit. It also independently checks modular targets, scale
restoration, partial report intervals and a short final minibatch. Diagnostic
trial manifests record actual training rows, updates, scale offsets, optimizer
settings and schedules, in addition to the original metrics and checkpoints.

`--bias-init uniform` initializes biases uniformly in
`[-scale/sqrt(fan_in), scale/sqrt(fan_in)]`, with `--bias-scale` defaulting to 1.
At the default bias scale of 1, `--bias-init domain` puts each first-layer hinge
through a random point
in the inner 90% of the operand box, using the actual input scale and valid
divisor range. Later biases are identical to uniform mode. Independent
per-layer streams keep all Kaiming matrices paired across every bias mode.
The [paired bias screen](../experiments/local-mlp-bias-screen.json) fixes both
operations, three input/target scales, three bias modes and two learning rates
at width 128, depth 4 and zero weight decay. Its 36 seed-11 runs each use 1,000
complete epochs and cosine decay. Native CLI initialization checks are in
`scripts/verify-mlp-initialization.mjs`.

Additional concrete grids can use the same checked local driver:

```sh
node scripts/local-mlp-diagnostics.mjs DIAGNOSTIC_BINARY target/local-ablations followup --trials experiments/local-mlp-bias-screen.json --label bias-screen
```

Each follow-up file records a reason and explicit trial settings before launch;
the source fingerprint includes its hash. Trial identities include the batch,
bias mode, decay schedule, epoch budget and seed, allowing matched optimizer
and capacity comparisons without changing completed experimental protocols.

`examples/scalar-relu-arithmetic.rs` is a separate representability control.
It constructs a plain raw-input W128 D8 network for division and multiplication
mod 97 and checks both complete operation tables on CUDA. The construction
interpolates the inverse divisor, multiplies through integer square identities,
and reduces via ReLU sawtooth hinges. It uses hand-set weights and does not run
an optimizer; it must never be presented as a Kaiming-trained accuracy result.

The combined investigation audit can also run while additional trials are
pending. It writes explicitly marked partial reports, validates the recorded
predictions and paired initializations, checks the Cartesian grid coverage,
and independently applies the prescribed finalist selection rules:

```sh
node scripts/summarize-mlp-investigation.mjs target/local-ablations --partial
```

After every planned phase and selected confirmation is complete, omit
`--partial` to require complete evidence and generate the final report:

```sh
node scripts/summarize-mlp-investigation.mjs target/local-ablations
uv run --locked scripts/plot-local-ablations.py target/local-ablations
```

The full-batch, width and optimizer comparisons have separate concrete plans in
`experiments/local-mlp-fullbatch-pilot.json`,
`experiments/local-mlp-capacity-pilot.json` and
`experiments/local-mlp-optimization-ablations.json`. Batch comparisons retain
the same number of complete epochs and visits per equation; full-batch and
minibatch update counts differ and are reported explicitly. Runtime measurements
share a busy local GPU and are not isolated speed benchmarks.

The [selected confirmation plan](../experiments/local-mlp-selected-confirmations.json)
records fixed training recipes before their seed-17/23 runs and reuses completed
seed-11 trials with checkpoint, initializer and history hashes. The combined
audit checks that their settings match and reports each seed separately.
Each seed changes the data partition, initialization and training order together;
variation across these runs does not isolate initialization from the data split.

`examples/scalar-mlp-evaluate.rs` reloads a completed model through the checked
parameter-loading API, recreates its seeded train/held-out split and exports
both prediction tables. It refuses an existing output directory. The independent
Node audit checks modular targets, split completeness and disjointness, all
reported metrics, the original held-out predictions and unchanged input hashes:

```sh
cargo oxide build --arch sm_86 --cargo-target-dir target/local-ablations/build -- --release --locked --example scalar-mlp-evaluate
node scripts/verify-mlp-checkpoints.mjs target/local-ablations/build/release/examples/scalar-mlp-evaluate target/local-ablations/checkpoint-verification-selected RUN_DIRECTORY...
```

The final combined report requires these checkpoint replays for every completed
model in the selected three-seed cohorts. This verifies actual saved weights in
addition to the training logs; it does not rerun optimization.
