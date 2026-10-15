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
