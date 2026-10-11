# Modular division size search

Search dense GPTMini first in
`/home/misha/lean_projects/transformer/experiments/mod97_size_search`, then
repeat with this repository's Mesh GPT. The user limits each run to 10,000
epochs, where an epoch visits every training equation once, and requests
frequent commits. Model size means learned parameter count, counting tied
tensors once. The user's successful model was around 500,000 parameters.

Both stages use all 9,312 valid `a / b mod 97` equations and fixed 50%
training splits. Batch 512 has nine full batches and a short batch of 48:
ten updates per epoch. The dense and Mesh implementations have different
seeded splits, token vocabularies, and supervised positions. These searches
therefore compare sizes within each architecture, rather than claiming an
otherwise matched cross-architecture experiment.

The dense starting recipe comes from the user's archived experiments:
AdamW, learning rate 0.001, weight decay 0.1, betas (0.9, 0.98), and ten
warmup updates. Dense GPTMini uses learned tied embeddings, fused QKV,
QKNorm with learned head temperatures, RoPE, XSA, parameter-free RMSNorm,
and a ReLU-squared feed-forward network. Mesh GPT retains its generated
embeddings, mesh projections, ELU, fixed QK normalization and tied head.

## Evidence and criteria

Evaluate both complete splits every 250 updates, or 25 epochs. Generalization
is the first observation with held-out accuracy at least 99%. On 2026-10-11
the user also made it the stopping rule: new runs pass
`--stop-validation-accuracy 0.99` and end at that observation. It replaced
20 consecutive observations at 99% on both splits, and it does not measure
whether accuracy stays above the target. Earlier runs trained their whole
budgets; their summaries apply the new rule to the recorded histories.
Delayed grokking additionally requires an earlier stretch of five
consecutive observations with training accuracy at least 99% and held-out
accuracy at most 10%. Record the first training 99% crossing separately.
This checks sampled observations, not every intervening update. A minimum
refers to tested configurations and seeds; failure of one setting does not
establish a universal lower bound.

Raw artifacts remain in ignored run directories. Summaries retain settings,
epoch counts and history SHA-256 hashes. `--output` also archives portable
accuracy histories beside the summary so plots do not depend on the original
run directories. Recompute them with:

```sh
uv run --locked scripts/summarize-grokking.py \
  /home/misha/lean_projects/transformer/experiments/mod97_size_search/runs/graph-w128 \
  target/grok-div97-search/default-w128-h4-l2-d4-f6-s7 \
  --output target/grok-div97-search/summary.json
uv run --locked scripts/plot-grokking-search.py \
  target/grok-div97-search/summary.json --output target/grok-div97-search/plots
```

The dense environment is locked to PyTorch 2.14.1+cu126. Its historical
CPU fixtures were generated with 2.14.0+cu126 and fail exact floating-point
comparisons under the current environment, including at update zero. The
new study does not change its training engine or fixtures. Fresh results
record their own engine fingerprints and do not claim bitwise equality
with those historical trajectories.

## Results

The preliminary Mesh control began before the user specified dense-first
ordering. It completed the full budget; subsequent Mesh searches follow
the dense stage. This control reported every 100 epochs; new search runs
report every 25 epochs.

| Model | Parameters | Width / blocks / heads | Mesh depths | LR / decay | Epochs | Final training / held-out | Result |
| --- | ---: | --- | --- | --- | ---: | --- | --- |
| Mesh control, seed 7 | 12,288 | 128 / 2 / 4 | 4 / 6 | 0.001 / 1 | 10,000 | 1.63% / 0.71% | Neither fitting nor grokking |
| Mesh control, seed 7, gain 1.4 | 12,288 | 128 / 2 / 4 | 4 / 6 | 0.001 / 0.1 | 10,000 | 6.23% / 1.80% | Neither fitting nor grokking |
| Mesh calibrated, seed 7, gain 1.4, embedding gain 0.1 | 26,624 | 128 / 2 / 4 | 8 / 16 | 0.003 / 0.01 | 10,000 | 32.28% / 2.08% | Neither fitting nor grokking |
| Mesh calibrated, seed 7, gain 1.4, embedding gain 0.1, Q/K signs | 26,624 | 128 / 2 / 4 | 8 / 16 | 0.003 / 0.01 | 10,000 | 27.75% / 1.89% | Neither fitting nor grokking |
| Mesh fitting pilot, seed 7, gain 1.4, embedding gain 0.02, Q/K signs | 507,904 | 1024 / 5 / 8 | 8 / 16 | 0.001 / 0.01 | 400 (retired) | 100% / 2.62% | Confirmed memorization plateau; no grokking before retirement |
| Dense GPTMini, seed 1 | 423,816 | 128 / 2 / 4 | Dense | 0.001 / 0.1 | 10,000 | 100% / 100% | Generalized at epoch 2,600 |
| Dense GPTMini, seed 1 | 113,608 | 64 / 2 / 4 | Dense | 0.001 / 0.1 | 10,000 | 100% / 100% | Generalized at epoch 100 |
| Dense GPTMini, seed 1 | 32,232 | 32 / 2 / 4 | Dense | 0.001 / 0.1 | 10,000 | 99.81% / 96.41% | Below held-out target at the cap |
| Dense GPTMini, seed 1 | 9,976 | 16 / 2 / 4 | Dense | 0.001 / 0.1 | 10,000 | 61.96% / 37.26% | Neither fitting nor grokking |
| Dense GPTMini, seed 1 | 32,232 | 32 / 2 / 4 | Dense | 0.001 / 0.2 | 10,000 | 100% / 100% | Delayed grokking at epoch 7,675 |
| Dense GPTMini, seed 1 | 19,568 | 24 / 2 / 4 | Dense | 0.001 / 0.2 | 10,000 | 99.89% / 37.46% | Fitted training; failed held-out target |
| Dense GPTMini, seed 1 | 9,976 | 16 / 2 / 4 | Dense | 0.001 / 0.2 | 10,000 | 69.61% / 34.60% | Neither fitting nor grokking |

The dense run first fitted training at epoch 725 and first reached 99%
held-out accuracy at epoch 2,600, 1,875 epochs later. It did not have the
strict near-chance memorization plateau. Having trained the whole budget,
it later fell below 99% at some observations; two in its final 1,000 epochs
did so, with minimum held-out accuracy 96.28%.

Width 64 first fitted training at epoch 75 and reached 99% held-out accuracy
at epoch 100. It did not exhibit the strict near-chance memorization plateau.

Width 32 first fitted training at epoch 575 but never reached 99% held-out
accuracy within the cap. Its late generalization transition reached 96.41%
held-out accuracy at epoch 10,000. Width 16 remained below the training
threshold. Changing only weight decay to 0.2 brought width 32's transition
inside the budget. Training first reached 99% at epoch 825. A near-chance
memorization plateau began at epoch 1,050 and was confirmed at 1,150.
Held-out accuracy first reached 99% at epoch 7,675, and the run finished at
100% on both splits. This 32,232-parameter model is the smallest tested
delayed-grokking model so far. Width 24 with the same stronger-decay recipe
finished at 99.89% training and 37.46% held-out accuracy, without a
generalization event.
Width 16 with decay 0.2 finished at 69.61% training and 34.60% held-out
accuracy. The dense stage completed seven fresh runs. The successful
recipe and refinements are committed as `033597a` in the dense repository;
the Mesh stage is active.

## Mesh search settings

The frozen candidates are in
[div97-mesh-search.json](../experiments/div97-mesh-search.json). The first
candidate has 495,616 parameters: width 512, two blocks, eight heads, and
attention/embedding and feed-forward mesh depths both 44. Initialization
gain 1.4 scales every original seeded mesh edge before training, including
each projection's final layer. It preserves all operators and connectivity.
A two-update smoke test had finite complete-split training loss
`15.23044 -> 13.28423`; this checks initialization and updates, not grokking.

The runner saves the exact command, frozen settings, binary SHA-256 and
console log. It skips a completed identical configuration and refuses to
overwrite a different or unfinished run. Mesh checkpoints currently hold
weights only, so an interrupted run needs a separately named fresh attempt.
Each runner invocation executes its selected candidates in order:

```sh
uv run --locked scripts/run-mesh-search.py experiments/div97-mesh-search.json --dry-run
uv run --locked scripts/run-mesh-search.py experiments/div97-mesh-search.json \
  w512-h8-l2-d44-f44-g14-wd01
```

Optional buffer reuse is enabled in
[div97-mesh-cached-search.json](../experiments/div97-mesh-cached-search.json).
Its separately named candidates use the same mathematics, seeds, batches
and budgets, with a bounded pool of three batch shapes. A 50-epoch,
500-update comparison at 123,904 parameters took 25.41 seconds with the
original binary and 18.85 seconds with reuse, a 1.35x speedup. All recorded
losses, accuracies and final checkpoint bytes matched exactly. The other
experiment was paused during timing. This is a timing experiment, not a
grokking result; speedup at other sizes is not established. Commands and
binary/checkpoint hashes are in
[workspace-cache-comparison.json](grokking-results/workspace-cache-comparison.json).
The GPU verification suite also checks exact AdamW trajectories across
full/tail batches, different sequence lengths, cache eviction, invalid
inputs and shrinking the cache. Independent numerical checks still pass.

The lightweight 12,288-parameter control overlapped the initial large-model
run and finished the full budget without fitting training data. Accuracy and
complete epochs are the comparison measures for overlapping runs. The large
control is temporarily paused with its optimizer state preserved in process
memory while the smaller calibrated comparisons run. Its pause is recorded
in the raw run's `scheduling.json`; it has no final-budget result yet.

Initialization refinements are in
[div97-mesh-calibrated-search.json](../experiments/div97-mesh-calibrated-search.json).
The initial 495,616-parameter run had cross-entropy 15.23, well above the
uniform 99-token baseline of about 4.60. The new `--embedding-output-gain`
scales only the final linear embedding layer after the global gain, setting
both the input and tied output embedding amplitudes. Defaults preserve the
original initialization. The candidates test gains 0.1 and 0.02, shallower
mesh projections, learning rates 0.001 and 0.003, and decay 0.01. Results
apply to each complete configuration because several settings vary.

The four original candidates passed a two-update, batch-eight initialization check,
with exhaustive evaluations and finite losses. Starting training losses
were 4.6310 (26,624 parameters), 4.5964 (63,488), 4.6765 (495,616), and
4.6417 (507,904). Zero, negative, NaN and infinite output gains are rejected
before device creation. Exact settings, commands, losses and the binary
hash are archived in
[mesh-initialization-smoke.json](grokking-results/mesh-initialization-smoke.json).
These checks establish valid execution; the full epoch-budget runs measure
learning and grokking. The first calibrated search used 26,624 parameters
and completed all 10,000 epochs without fitting the training set. Its final
training/held-out cross-entropies were 2.7389/6.7946, and neither split
reached 99%. Its controlled sign-initialization variant also completed the
full budget without either split reaching 99%, finishing at 27.75% training
and 1.89% held-out accuracy. The first-block initialization geometry check
therefore did not produce a successful small model under these settings.
The five-block, 63,488-parameter variant is running. A 507,904-parameter sign variant uses
batch 256: eighteen full batches and the short batch of 48, or nineteen
updates per epoch. It evaluates every 475 updates (25 epochs) and has a
190,000-update cap. Its two-update capacity check passed with finite
complete-split losses; that check is excluded from the grokking results.

The controlled `--qk-signs 1` variant changes only the final Q/K rows using
a separate seeded sign stream. An independent f64 reconstruction of the
initialized first block gives mean self-attention probability 82.83% with
the original directions and 29.82% with the signs, averaged over all 9,312
inputs and four heads. The captured parameter magnitudes are exactly equal,
and every other parameter is unchanged. Capture uses one update at learning
rate `1e-30`; observed losses are unchanged. This checks initialization
geometry, not learning. The complete-budget controlled run measures whether
this difference helps fitting and grokking.

See [qk-initialization-analysis.json](grokking-results/qk-initialization-analysis.json).
Reproduce the captures and f64 analysis in a fresh directory:

```sh
uv run --locked scripts/analyze-mesh-qk-initialization.py --output target/qk-init-recheck
```

The regularization candidates are frozen in
[div97-mesh-regularization-search.json](../experiments/div97-mesh-regularization-search.json).
They retain the five-block signed initialization and batch-256 protocol,
increase weight decay to 0.1, and test widths 1024, 512, 256 and 128
(507,904, 253,952, 126,976 and 63,488 parameters). Width 1024 is running;
the smaller three have not run yet.
At width 1024 only decay changes from the retained fitting pilot.
To prioritize regularization trials, that decay-0.01 run may be retired
after it establishes five consecutive training-99% / held-out-at-most-10%
observations, or after reaching the 500-epoch fitting-pilot budget. The
500-epoch stop was added after the first transient training-99% result at
epoch 200, before observing epoch 500. Any retired run is recorded as
interrupted at its last
observation, without claiming failure over the full 10,000-epoch budget.
The stronger-decay trial starts from the original seed, without reusing
weights or optimizer state. Each fresh trial remains capped at 10,000 epochs.

The 507,904-parameter pilot first reached training 99% at epoch 200, then
fell below the threshold at epochs 225, 250 and 275. A qualifying
memorization plateau began at epoch 300 and was confirmed at epoch 400;
training was 100% at the last four observations, with final held-out
accuracy 2.62%. The pilot was retired at its epoch-400 checkpoint. This
result covers 400 epochs and leaves later generalization under that recipe
unmeasured. Its retained checkpoint hash, scheduling events and the fresh
run's command are in
[mesh-regularization-handoff.json](grokking-results/mesh-regularization-handoff.json).
The binary hash, complete split bytes and every initial recorded loss,
accuracy and weight norm match between the pilot and fresh trial. The sole
settings difference is weight decay, 0.01 to 0.1.

Smaller-model head pilots are defined in
[div97-mesh-head-pilots.json](../experiments/div97-mesh-head-pilots.json).
They keep 63,488 learned mesh parameters, width 128, five blocks and depths
8/16, and vary heads between two, one and four. The two-head pilot started
after the 26,624-parameter signed control finished. These pilots
use exactly 500 complete epochs and measure early fitting. Their shorter
budgets cannot establish failure to grok within 10,000 epochs. A promising
pilot gets a separately named full-budget run from the original seed.
The eight-head full run supplies the corresponding first-500-epoch control
with the same split, seed, initialization, batches, learning rate and decay.

The two-head pilot completed 500 epochs at 20.15% training and 1.91%
held-out accuracy, without reaching training 99%. The eight-head control
was at 23.67% training and 2.04% held-out accuracy at the same epoch.
These observations do not favor fewer heads for early fitting at this size.
One- and four-head pilots remain unrun.

Depth pilots are defined in
[div97-mesh-depth-pilots.json](../experiments/div97-mesh-depth-pilots.json).
They use width 128, two heads, two blocks, embedding-output gain 0.1 and
attention/embedding depths 16, 32 and 64, with feed-forward depths twice
as large. Their counts are 53,248, 106,496 and 212,992 parameters, and
each pilot visits the full training set 500 times. The depth-32 pilot is
selected first. These vary the complete configuration from the head
pilots; their results cannot isolate a single cause or establish failure
at the 10,000-epoch cap.

The depth-32 pilot completed 500 epochs at 11.71% training and 2.13%
held-out accuracy. The fresh matched pilot changed only learning rate from
0.001 to 0.003 and finished at 3.41% training and 3.01% held-out accuracy.
Its binary, split and every initial recorded loss, accuracy and weight norm
match the first pilot. Neither reached training 99%; these short results
do not establish either recipe's behavior over 10,000 epochs. The other
depth variants have not run yet. Pilot summaries are separate from
the full-budget comparison in
[head-pilot-summary.json](grokking-results/head-pilot-summary.json),
[head-pilot-comparison.json](grokking-results/head-pilot-comparison.json),
and [depth-pilot-summary.json](grokking-results/depth-pilot-summary.json).
The matched learning-rate evidence is in
[depth-pilot-comparison.json](grokking-results/depth-pilot-comparison.json).

See the [portable results](grokking-results/summary.json) and
[learning curves](grokking-results/grokking-size-search.svg). The original
Mesh defaults are substantially
smaller than the user's successful parameter count, so their failure does
not settle the larger-model search.

## Architecture ablations

Before these ablations, no completed Mesh run raised held-out accuracy above
chance, including the runs that fit the training set. Two architectural
constraints could explain this. The embedding mesh computes each token vector
from fixed codes of an abstract symbol ID through a depth-`d` neighbor mesh.
Channel `c` therefore depends only on code channels `c..c+d`, and tokens
cannot receive independent vectors. Neighbor meshes also mix channels only
locally, so the Q/K/V, output and feed-forward meshes cannot form global
channel combinations.

The factorial ablation in
[div97-mesh-ablations.json](../experiments/div97-mesh-ablations.json) crosses
`--embedding mesh|table` with `--connectivity neighbor|butterfly`; see
[Mesh GPT](mesh-gpt.md#ablation-variants). All four arms use width 128, four
heads, two blocks, depths 8/16, initialization gain 1.4, embedding-output gain
0.1, and the dense recipe's learning rate 0.001 and decay 0.1. Mesh arms have
26,624 learned parameters, and table arms have 37,248. Each table arm starts as
the same function as its mesh arm, and their one-epoch smoke runs had identical
initial complete-split losses. Each arm has the 10,000-epoch cap.

A first launch without a stop rule reached epoch 3,325 in the mesh arms and
3,575 in the table arms before the stopping rule changed. Its table arms first
reached 99% held-out accuracy at epochs 375 (neighbor) and 525 (butterfly).
Afterwards, transient loss spikes lowered accuracy on both splits at once:
held-out accuracy fell below 99% at 7 of 129 and 12 of 123 later
observations, to minima of 85.72% and 76.91%. Its mesh arms stayed near
chance, peaking at 6.29% and 22.68% training and 2.38% and 2.36% held-out
accuracy. All four arms then restarted from the original seed with
`--stop-validation-accuracy 0.99`. The interrupted runs keep their
interruption records and checkpoints in their ignored directories.

The restarted arms ran concurrently until both table arms stopped. With one
GPU, concurrency does not raise total throughput, so the mesh arms then ran
one at a time. At epoch 375, mesh-butterfly was suspended with `SIGSTOP`
while mesh-neighbor finished, then resumed with `SIGCONT`. The suspended
process kept its weights and AdamW moments in memory. `scheduling.json` in
each mesh run directory records both signals, and recorded elapsed seconds
include the suspension. Every observation the restarted arms share with the
interrupted launch matches it exactly: 134 per mesh arm through epoch 3,325,
and all 16 and 22 table observations. The suspension therefore did not
change the mesh-butterfly trajectory.

| Arm | Parameters | First training 99% | First held-out 99% | Epochs run | Final training / held-out | Peak held-out |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Mesh embedding, neighbor | 26,624 | Not reached | Not reached | 10,000 | 5.48% / 1.37% | 2.38% |
| Table embedding, neighbor | 37,248 | Epoch 175 | Epoch 375 | 375 | 100% / 99.12% | 99.12% |
| Mesh embedding, butterfly | 26,624 | Not reached | Not reached | 10,000 | 27.64% / 1.87% | 2.38% |
| Table embedding, butterfly | 37,248 | Epoch 200 | Epoch 525 | 525 | 100% / 99.16% | 99.16% |

Both table arms generalized, 200 and 325 epochs after first fitting the
training set. Neither had the near-chance memorization plateau, so these are
generalization events rather than delayed grokking. Both mesh arms used the
whole budget, and their held-out accuracy never exceeded 2.38%, against 1/97
or 1.03% for a uniform guess. Butterfly strides raised the mesh arm's
training fit, to a peak of 28.63% at epoch 9,925 against 6.29% at epoch 400
with neighbor strides, but not its held-out accuracy. The strides apply to
every mesh, so in that arm each embedding channel also depends on every code
channel. Under this recipe a learned table was sufficient for
generalization. Global channel mixing was neither necessary (table-neighbor)
nor sufficient (mesh-butterfly). Each arm is a single seed, so the 150-epoch
gap between the table arms favors neither connectivity.

[analyze-token-vectors.py](../scripts/analyze-token-vectors.py) reads the
residue vectors each run saves at its end. Division subtracts discrete
logarithms, so the script orders the 96 nonzero residues by their discrete
logarithm base 5, a primitive root modulo 97. Along this cycle it measures
the share of non-constant Fourier power in the five strongest frequencies,
compared with 1,000 random cycle orders. It also reports the effective rank
of the centered vectors (the exponential of their spectral entropy), the
directions holding 99% of their energy, and cosine similarity between
residues. Results are in
[ablation-token-vectors.json](grokking-results/ablation-token-vectors.json),
and the training summaries in
[ablation-summary.json](grokking-results/ablation-summary.json):

| Residue vectors | Epoch | Top-5 Fourier share (null mean / 99th percentile) | p | Effective rank (99% directions) | Nearest-neighbor cosine, mean (maximum) |
| --- | ---: | --- | ---: | --- | --- |
| All four smoke runs | 1 | 0.117–0.120 (0.118–0.119 / 0.123–0.124) | 0.23–0.77 | 76.6–78.0 (92) | 0.415–0.428 (0.496–0.521) |
| Table, neighbor | 375 | 0.435 (0.143 / 0.158) | 0.001 | 30.0 (63) | 0.515 (0.628) |
| Table, butterfly | 525 | 0.492 (0.149 / 0.166) | 0.001 | 24.9 (61) | 0.465 (0.549) |
| Mesh, neighbor | 10,000 | 0.226 (0.203 / 0.247) | 0.100 | 5.0 (15) | 0.918 (0.979) |
| Mesh, butterfly | 10,000 | 0.200 (0.205 / 0.248) | 0.591 | 4.8 (5) | 0.912 (0.988) |

A p-value of 0.001 is the smallest that 1,000 permutations can report. Both
tables concentrate significant power in a few discrete-log frequencies:
17, 27, 33, 47 and 21 with neighbor strides, and 38, 45, 42, 48 and 6 with
butterfly strides. The two sets do not overlap. The generated vectors
instead collapse: effective rank falls from about 77 after one epoch to 5,
and residues become nearly parallel. With so few directions left, random
orders also concentrate more power, which raises their null, and their
observed shares stay within it. Because the head is tied, these vectors also
score every answer, so the head can separate answers only within the few
remaining directions.

Within this recipe, the generated token embedding is the binding
constraint. Replacing it with a table that starts as the same function was
enough to generalize, while global mixing in every mesh, the embedding mesh
included, was not. Its 2,048 shared weights generate all 99 vectors, so no
residue's vector can change independently. These arms do not separate the
responsible property: the number of shared weights, the two-parent layer
form, the fixed input codes, or tying the head to the generated vectors.
