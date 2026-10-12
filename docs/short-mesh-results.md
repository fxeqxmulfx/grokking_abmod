# Mesh models with short memorization plateaus

Allowing a short plateau exposes two useful candidates. The most consistent
sharp accuracy transition among the tested profiles is a **width-384,
depth-5 butterfly ReLU mesh MLP**. A **width-128, two-block, three-mesh-layer
Square residual mesh** is about three times smaller, with a slower transition.
The subsequently [verified Neighbor variant](neighbor-mesh-results.md) has
the same size and a simpler visualization; seed 11 has a 34-epoch
perfect-train plateau and a 241-epoch test rise.

These measurements use Rust/CUDA from initialization, division modulo 97,
4,656 equations per split, batches of 512 plus a 48-equation tail, AdamW with
betas (0.9, 0.98), learning rate 0.001, and ten warmup updates. Both complete
splits are evaluated **every epoch**, rather than every 25 epochs. Different
seeds change both initialization and the train/held-out split.

The relaxed rule requires a contiguous interval of at least 25 epochs at
train accuracy ≥99% and test accuracy ≤10%, then the first test ≥90%
observation within 250 epochs of the last ≤10% report, and a later ≥99%
test result. Reported lengths are sampled observations at one-epoch resolution.

## Replication across seeds

| Profile | Parameters | Seeds | Plateau length | Test ≤10% → ≥90% | First test ≥99% | Strict relaxed-rule matches |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Butterfly ReLU mesh W384, depth 5, decay 1 | 114,816 | 7–11 | 31–57 | 113–141 | 506–534 | **5/5** |
| Butterfly Square residual W128, B2 M3, decay 0.1 | 38,784 | 7–11 | 21–51 | 194–260 | 440–546 | **3/5** |
| Neighbor Square residual W128, B2 M3, decay 0.1 | 38,784 | 7–11 | 28–65 | 180–278 | 414–588 | **3/5** |
| Butterfly Square residual W96, B2 M3, decay 0.1 | 29,088 | 7–9 | 29–134 | 235–405 | 580–1,047 | **1/3** |
| Butterfly Square residual W64, B2 M3, decay 0.1 | 19,392 | 7–12 | Below the qualifying rise rule | Rise >250 epochs | Reached ≥99% | **0/6** |

For W128, seed 8 misses the minimum plateau by four epochs (21 instead of
25); seed 9 misses the maximum rise by ten (260 instead of 250). All five
reach ≥99% test. Those boundary misses differ from failing to learn.

For ReLU W384:

| Seed | Plateau interval | Plateau length | Last test ≤10% | First test ≥90% | Rise | First test ≥99% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 269–319 | 50 | 319 | 460 | 141 | 534 |
| 8 | 307–338 | 31 | 338 | 456 | 118 | 520 |
| 9 | 297–345 | 48 | 345 | 458 | 113 | 506 |
| 10 | 295–352 | 57 | 352 | 466 | 114 | 516 |
| 11 | 288–330 | 42 | 330 | 452 | 122 | 508 |

For Square residual W128:

| Seed | Plateau interval | Plateau length | Rise | First test ≥99% | Match |
| --- | --- | ---: | ---: | ---: | --- |
| 7 | 92–142 | 50 | 205 | 467 | Yes |
| 8 | 79–100 | 21 | 194 | 440 | Below the 25-epoch plateau minimum |
| 9 | 96–144 | 48 | 260 | 537 | Above the 250-epoch rise maximum |
| 10 | 83–111 | 28 | 221 | 488 | Yes |
| 11 | 97–148 | 51 | 229 | 546 | Yes |

The [multi-seed graph](grokking-results/short-plateau-mesh/short-mesh-seeds.png)
shows solid test curves and dashed train curves, with separate time axes
for each profile. Complete raw observations and metadata are in the
[archived summary](grokking-results/short-plateau-mesh/summary.json).

## What the plateau means

The table's train threshold is **99%**, not zero classification error or
near-zero cross entropy. Training confidence continues to improve through
these short intervals, and held-out loss can begin declining before held-out
accuracy rises sharply.

At the stricter **100% train / ≤10% test** threshold, ReLU W384 has only a
six-epoch interval on seed 9 and no multi-report interval on the other four
seeds. Square W128 has intervals of 22, 23, and 26 epochs on seeds 7, 9,
and 11 respectively; seeds 8 and 10 have none. The archived analysis records
both thresholds. For a fixed-seed butterfly demonstration with a
complete-memorization interval, Square W128 seed 11 has 26 epochs at perfect
train, then a 229-epoch test rise. The later Neighbor seed 11 reaches a
34-epoch perfect-train interval and a 241-epoch rise, with simpler local edges.
Its five independent runs are archived separately in the
[Neighbor verification](neighbor-mesh-results.md).

These are short accuracy plateaus, with smooth but comparatively steep
transitions. They are not evidence of a discontinuous jump or of a long
zero-error memorization phase. Compare the actual accuracy and loss curves:
[ReLU W384 seed 7](grokking-results/short-plateau-mesh/relu-mesh-short-plateau.png),
[Square W128 seed 7](grokking-results/short-plateau-mesh/residual-mesh-short-plateau.png).

## Architectures

The ordinary ReLU mesh uses a dense 194×384 input projection, four sparse
384-channel meshes, and a dense 384×97 readout. Every mesh neuron reads its
own channel and one other channel; the four butterfly strides are 1, 2, 4,
and 8, with circular indexing. Every hidden layer uses ReLU. There are no
biases or normalization parameters. Counts are 74,496 input weights,
3,072 intermediate mesh edges, and 37,248 readout weights: **114,816**.

The Square residual model uses a dense 194×128 input and two residual blocks.
Each branch is RMSNorm → linear mesh → Square → linear mesh → Square →
linear mesh, added to the branch input. Final RMSNorm precedes the 128×97
readout. Butterfly strides continue across branches: 1/2/4, then 8/16/32.
Counts are 24,832 input weights, 1,536 mesh edges, and 12,416 readout weights:
**38,784**. RMSNorm has no learned affine parameters.

Reducing the Square model to width 64 retains a visible delay, but the
transition becomes much slower. Reducing the ordinary ReLU model to width
128, 96, or 64 with the same decay does not preserve the qualifying
memorization plateau on the tested seed 7. Width alone is not a safe change
to the observed behavior.

## Reproduce and continue searching

The active loop prioritizes 22 replications and compact variants in
[`experiments/div97-short-plateau-mesh.args`](../experiments/div97-short-plateau-mesh.args),
then continues its reproducible exploration. The original long-plateau
study is preserved in its own directory and stopped to avoid concurrent
GPU trials. The short-plateau loop is separate:

```bash
cargo build --release --locked --bin sharp-mesh-search
target/release/sharp-mesh-search \
  --output target/grok-div97-search/short-plateau-mesh-loop \
  --trainer target/grok-div97-search/sharp-mesh-loop/bin/mesh-gpt \
  --plateau-epochs 25 --report-every 10 \
  --queue experiments/div97-short-plateau-mesh.args

# Archive completed trials and regenerate the figures.
UV_CACHE_DIR=/tmp/grokking-uv-cache uv run --locked scripts/analyze-short-mesh.py \
  target/grok-div97-search/short-plateau-mesh-loop \
  --output docs/grokking-results/short-plateau-mesh
```

The raw archives include exact argument arrays, run metadata, complete
loss/accuracy histories, outcome classifications, and checkpoint hashes.
Original coarse-cadence and new fine-cadence replays match exactly to the
saved precision at their common epochs: 19 shared observations for Square
seed 7 and 21 for ReLU seed 9. Loss, accuracy, and weight norms were checked;
timings are excluded. See [replay checks](grokking-results/short-plateau-mesh/replay-check.json).

The scheduler's nine tests and strict Clippy checks pass. New tests verify
that one-epoch observations confirm a 25-epoch interval, missing observations
break it, and prioritized queues reject non-mesh models, assisted operand
encodings, and output overrides. No neural kernels changed in this refinement.
