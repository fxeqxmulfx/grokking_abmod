# Shared Neighbor embedding and readout

This Rust experiment uses two incoming activation connections per learned
neuron, including the input embedding and the output chain. One set of linear
Neighbor layers is shared by operand a, operand b, and the transposed readout.
The readout uses those layers in reverse order with reversed connections; it
is exactly the implicit embedding's transpose. No dense E matrix is stored.

The selected configuration is width 128, 32 shared embedding layers, and 64
pre-RMSNorm Neighbor + ReLU residual blocks. Both operands use ordinary one-hot
indicators for residues 0–96, padded with zeros to 128 channels. The b embedding has a fixed
adjacent-pair Rotate90 role; its result is added to the a embedding. The core
has one bias per neuron. Each branch normalizes the incoming stream, applies
its two-parent affine layer and ReLU, and adds the result to the unchanged
stream. A final RMSNorm precedes the readout. All RMSNorms are parameter-free
with epsilon 1e-5. Embedding/readout chains remain linear, without biases or
activation. The first 97 terminal values are the answer logits.

| Parameter group | Stored parameters |
| --- | ---: |
| 32 shared Neighbor embedding layers | 8,192 |
| 64 core layers, two weights per neuron | 16,384 |
| Core biases | 8,192 |
| Total | 32,768 |

Each embedding edge has one optimizer state, and its gradient sums its three
uses. The embedding chain starts at identity plus independent Gaussian neighbor
mixing with standard deviation `0.1/sqrt(32)`. The first core layer uses
Kaiming-normal two-parent weights; the other 63 start at identity and learn
their second-parent weights. All biases start at zero.

The depth check propagates sets of input identity routes through the actual
Neighbor links, Rotate90, the sum, the core and the transposed readout. Every
one of the 97 logits reaches all 194 a/b indicators. This includes b=0's unused
indicator; division training uses b=1–96. This is potential connectivity, not
a guarantee that learned weights or ReLU gates will carry every dependency.
The depth check deliberately excludes the additional global dependencies of
RMS statistics, so full coverage also holds through the learned value routes.

The [complete SVG](diagrams/shared-neighbor-embedding-readout.svg) shows all
54,144 nodes and 99,263 connections, including fixed RMS trees and residual
links. S1–S32 identify shared weights, N the pre-RMSNorms, M1–M64 the core, + the
residual sums, Final N the final RMSNorm and R the transposed readout. Each norm
is represented by unary squares, a binary sum tree, a fixed RMS scale and
two-parent normalized outputs. GPU warp reductions implement the same RMS
function; the drawn reduction tree does not prescribe FP32 summation order.
Generate the graph from Rust with:

```bash
cargo run --locked --bin full-mesh-svg -- docs/diagrams
```

The [queued arguments](../experiments/div97-shared-neighbor.args) contain only
this architecture and depth. The earlier full-mesh binary-input study is
stopped; its completed and interrupted artifacts remain in
`target/grok-div97-full-mesh/`.

```bash
cargo oxide build --arch sm_86 -- --release --locked --bin mesh-gpt --bin sharp-mesh-search
target/release/mesh-gpt grok-simple --verify-model shared-neighbor-mesh-mlp
target/release/sharp-mesh-search \
  --output target/grok-div97-shared-neighbor-prenorm \
  --queue experiments/div97-shared-neighbor.args \
  --epochs 1000 --report-every 10 --max-trials 1 \
  --plateau-epochs 25 --jump-epochs 250
```

Training uses division modulo 97, seed 11, a 50/50 split, minibatches of 512
with a 48-row tail, and complete-split evaluation every epoch. AdamW has
learning rate 0.001, weight decay 0.1, betas 0.9/0.98, epsilon 1e-8 and a
10-update warmup. Each pilot has a 1,000-epoch budget and an early stop at test
>=99%. The unchanged grokking criterion requires at least 25 elapsed epochs
with train >=99% and test <=10%, then a <=250-epoch rise from the last test
<=10% observation to test >=90%, followed by test >=99%.

The corrected identity-hidden pilot completed all 1,000 epochs in 320.6 seconds.
Final train accuracy was **6.1426%**, held-out accuracy **1.8256%**, and cross
entropy **4.3383 / 4.8668**. Peak train accuracy was 6.4433% at epoch 775;
peak held-out accuracy was 2.4055% at epoch 1. It did not memorize the training
split or satisfy any grokking interval. This is a negative result for the
specified initialization and optimizer on seed 11, not evidence that other
settings or longer runs cannot learn.

The [archived measurements](grokking-results/shared-neighbor/pre-norm/summary.json)
include every epoch, run metadata and trainer/checkpoint hashes. The
[accuracy and loss plot](grokking-results/shared-neighbor/pre-norm/shared-neighbor-w128.svg)
shows both complete splits. The experiment remains Rust-only.

A control uses the same topology and optimizer with Kaiming-normal weights in
all 64 core blocks, instead of only the first block. Its
[queued arguments](../experiments/div97-shared-neighbor-kaiming.args) change
only `--mesh-initialization`; its study is
`target/grok-div97-shared-neighbor-prenorm-kaiming/`. At epoch 1,000 it reached
**5.0902% train / 1.7397% test**, with cross entropy **4.3996 / 4.7561**. It also
failed to memorize or grok. The
[archived control](grokking-results/shared-neighbor/pre-norm-kaiming/summary.json)
and [loss plot](grokking-results/shared-neighbor/pre-norm-kaiming/shared-neighbor-w128.svg)
preserve the measurements. The
[initializer comparison](grokking-results/shared-neighbor/initializer-comparison.json)
confirms identical shared embedding weights, first core block, trainer binary
and training settings; only core blocks 2–64 start differently.

The [zero-decay control](../experiments/div97-shared-neighbor-no-decay.args)
keeps the baseline initializer and changes only weight decay from 0.1 to zero.
Its study is `target/grok-div97-shared-neighbor-prenorm-no-decay/`. It completed
1,000 epochs with **6.5722% train / 1.6538% test**, cross entropy
**4.3328 / 5.1384**, and no memorization or grokking. The
[archive](grokking-results/shared-neighbor/pre-norm-no-decay/summary.json) and
[loss plot](grokking-results/shared-neighbor/pre-norm-no-decay/shared-neighbor-w128.svg)
record this control. Removing decay improved the minimum embedding norm to
0.2154 and reduced the maximum to 96.3827, but did not restore training accuracy.

Saved-checkpoint diagnostics measure the actual implicit embedding, activations
on the first 512 seeded training equations, and f64 parameter gradients on the
first 16. Their logits match the independent f64 reference exactly on eight
rows for both initial and final checkpoints. Every core block receives a
nonzero gradient on this diagnostic batch. Intermediate activations retain
substantial variation between examples; total hidden-state collapse is not
supported by these measurements.

The embedding remains localized: mean energy participation is 1.0206 effective
channels initially, then 2.2834 with identity-hidden or 2.1409 with Kaiming, out
of 128 channels. Without decay it remains 2.3070. This statistic is squared
total coordinate energy divided by the sum of squared coordinate energies;
it is not a count of nonzero weights.
Final class embedding norms range from **0.002205 to 411.166** for identity-hidden
and **0.0001424 to 252.154** for Kaiming. Pre-RMSNorm normalizes the core stream;
it does not constrain the 32-factor shared linear operator or equalize class
readout scales. Unequal operand scales can also make one operand weak in the
joined input. This suggests a conditioning and mixing problem, rather than
establishing one causal explanation.

All three pilots predict every a=0 equation correctly. For a!=0, identity-hidden
reaches **5.2472% train / 0.7168% test**, and Kaiming **4.1847% / 0.6299%**.
The no-decay control reaches **5.6808% / 0.5430%** on these nonzero inputs.
Class 0's large embedding dominates raw-logit direction statistics, so the
diagnostics additionally center class logits and inspect nonzero operands and
classes separately. Softmax ignores common shifts of class logits.

The [diagnostic summary](grokking-results/shared-neighbor/diagnostic-summary.json)
includes native predictions, checkpoint hashes and the measured traces. Run
the checked Rust analysis on any of the three trial directories with:

```bash
cargo run --release --locked --bin shared-neighbor-diagnostics -- \
  target/grok-div97-shared-neighbor-prenorm/trials/000001-shared-neighbor-mesh-mlp-w128-d64-s11-a01 \
  docs/grokking-results/shared-neighbor/pre-norm/diagnostics
node scripts/summarize-shared-neighbor.mjs
```

The mixing hypothesis has a matched core-connectivity control in
[these arguments](../experiments/div97-shared-neighbor-core-mixing.args).
The shared Neighbor embedding/readout remains unchanged; only core second
parents use repeating strides 1,2,4,8,16,32,64 instead of stride 1. Width,
depth, parameter count, seed, optimizer and complete initial parameters are
identical. Blocks 2–64 initially have zero second-parent weights, so the initial
functions also match. The
[paired-control checks](grokking-results/shared-neighbor/core-mixing/paired-control.json)
confirm identical checkpoint hashes, trainer, initial complete-split metrics
and initial full-table interaction measurements.

At epoch 1,000, this control reaches **7.3239% train / 1.5679% test**, with cross
entropy **4.2662 / 4.9718**. The corresponding Neighbor baseline reaches
6.1426% / 1.8256%. The
[archive](grokking-results/shared-neighbor/core-mixing/summary.json) and
[learning curves](grokking-results/shared-neighbor/core-mixing/shared-neighbor-core-mixing-w128.svg)
record the measurements. Longer core links did not restore memorization in
this seed-11 pilot. This is a control experiment, not a change to the selected
Neighbor model or the website.

The [interaction diagnostic](../src/bin/shared-neighbor-interactions.rs)
evaluates the independent f64 reference on all 9,216 nonzero operand pairs.
It excludes the trivial a=0 case and answer class 0, centers each row's nonzero
class logits, and decomposes every class score into a grand mean, separate
a and b effects, and a joint residual. The reported fraction is the mean
within-class proportion of score variance attributable to that joint residual.
Five tests cover additive grids, XOR, constant grids, a known mixed fraction
and unequal output scales. The complete ideal division indicator table gives
100% joint variance; it is a validation target, not a universal threshold for
the logits of an accurate model.

| Configuration | Initial joint fraction | Trained joint fraction | Train accuracy |
| --- | ---: | ---: | ---: |
| Neighbor core | 15.6766% | 36.4852% | 6.1426% |
| Distant-parent core | 15.6766% | 39.1561% | 7.3239% |

The trained diagnostic's nonzero-grid accuracy agrees with the archived CUDA
predictions. Full per-class and distance-group measurements are in each
trial's `diagnostics/joint-interaction*.csv`. RMSNorm itself creates joint
dependence through a shared input-dependent scale, so joint variance alone
does not establish useful feature mixing or arithmetic computation. These
results do not confirm that short core links are the sole cause of underfitting.
The shared embedding chain was subsequently tested with the paired initialization
control below, holding initial class norms and core weights fixed.

Run the full-table analysis on the baseline and the core control with:

```bash
cargo run --release --locked --bin shared-neighbor-interactions -- \
  target/grok-div97-shared-neighbor-prenorm/trials/000001-shared-neighbor-mesh-mlp-w128-d64-s11-a01 \
  docs/grokking-results/shared-neighbor/pre-norm/diagnostics
cargo run --release --locked --bin shared-neighbor-interactions -- \
  target/grok-div97-shared-neighbor-core-mixing/trials/000001-shared-neighbor-mesh-mlp-w128-d64-s11-a01 \
  docs/grokking-results/shared-neighbor/core-mixing/diagnostics
```

The corrected implementation passed 81 Rust tests, strict Clippy and the locked
CUDA release build. GPU logits, loss, parameter gradients and offset AdamW
updates were checked against the independent f64 reference, including the
actual width-128 / embedding-depth-32 / core-depth-64 configuration. Its maximum
relative gradient discrepancy was 3.2e-6. The
[verification log](grokking-results/shared-neighbor/pre-norm/validation/grokking-shared-neighbor-prenorm-cuda-verification.log)
also records gathered batches, repeatability and invalid launch/input checks.

The [initial run without normalization](grokking-results/shared-neighbor/no-norm/summary.json)
is archived separately. Its missing RMSNorm was an implementation error; its
2.86% train / 1.55% test result is not a measurement of the corrected model.

## Paired embedding mixing control

Six completed 1,000-epoch runs compare weak and strong mixing in the same
32-layer shared Neighbor embedding/readout and 64-block pre-norm Neighbor core.
The 32 existing embedding matrices form 16 paired rotations with cyclic shifts.
Weak pairs use an angle of approximately 1.013 degrees; strong pairs use signed
45-degree rotations. Both initialize every implicit class embedding to unit
norm and have Gram energy rank 97. Core weights are byte-identical within each
seed pair; the connectivity, 32,768 parameters, optimizer and data split are
unchanged. The rotation and norm constraints apply only at initialization.

| Seed | Weak train | Strong train | Weak test | Strong test |
| --- | ---: | ---: | ---: | ---: |
| 11 | 6.5722% | 6.4863% | 1.4820% | 1.5034% |
| 17 | 6.2715% | 6.5077% | 1.6323% | 1.4175% |
| 23 | 6.1211% | 6.9588% | 1.4175% | 1.5464% |

Stronger mixing spreads initial embedding energy across more coordinates but
does not restore memorization or generalization on these three seeds. This
does not identify the sole cause of underfitting. It rules out stronger initial
mixing as a sufficient fix under this protocol; it does not test a constrained
orthogonal embedding throughout training.

The [paired report](grokking-results/shared-neighbor/embedding-mixing/paired-summary.json)
verifies initial norms, rank, identical core weights, matching CPU/GPU initial
checkpoints and archived histories. The [full archive](grokking-results/shared-neighbor/embedding-mixing/summary.json)
contains all six runs and their source hashes. Validation passed 88 Rust tests,
strict Clippy, the CUDA build and all seven GPU/f64 verification fixtures.
No additional sparse search was started after the user's switch to the Dense
frontend plan.
