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
requires training and held-out accuracy at least 99% for 20 consecutive
observations. Delayed grokking additionally requires an earlier stretch of
five consecutive observations with training accuracy at least 99% and
held-out accuracy at most 10%. Record the first 99% crossing separately.
Stability at the budget requires every observation in the final 1,000 epochs
to meet 99% on both splits. This checks sampled observations, not every
intervening update. A minimum refers to tested configurations and seeds;
failure of one setting does not establish a universal lower bound.

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
| Dense GPTMini, seed 1 | 423,816 | 128 / 2 / 4 | Dense | 0.001 / 0.1 | 10,000 | 100% / 100% | Sustained generalization; final window not stable |
| Dense GPTMini, seed 1 | 113,608 | 64 / 2 / 4 | Dense | 0.001 / 0.1 | 10,000 | 100% / 100% | Sustained generalization; final window stable |
| Dense GPTMini, seed 1 | 32,232 | 32 / 2 / 4 | Dense | 0.001 / 0.1 | 10,000 | 99.81% / 96.41% | Below held-out target at the cap |
| Dense GPTMini, seed 1 | 9,976 | 16 / 2 / 4 | Dense | 0.001 / 0.1 | 10,000 | 61.96% / 37.26% | Neither fitting nor grokking |
| Dense GPTMini, seed 1 | 32,232 | 32 / 2 / 4 | Dense | 0.001 / 0.2 | 10,000 | 100% / 100% | Delayed grokking; final window stable |
| Dense GPTMini, seed 1 | 19,568 | 24 / 2 / 4 | Dense | 0.001 / 0.2 | 10,000 | 99.89% / 37.46% | Fitted training; failed held-out target |
| Dense GPTMini, seed 1 | 9,976 | 16 / 2 / 4 | Dense | 0.001 / 0.2 | 10,000 | 69.61% / 34.60% | Neither fitting nor grokking |

The dense run first fitted training at epoch 725 and first reached joint
99% accuracy at epoch 2,600. Twenty consecutive qualifying observations
began at epoch 3,675 and were confirmed at epoch 4,150, a 2,950-epoch lag
after training fit. It did not have the strict near-chance memorization
plateau. Two observations in its final 1,000 epochs fell below 99%, with
minimum held-out accuracy 96.28%.

Width 64 first fitted training at epoch 75 and reached joint 99% at epoch
100. Its first 20-observation qualifying stretch began at epoch 1,375 and
was confirmed at epoch 1,850, following earlier transient failures. Every
observation in its final 1,000 epochs exceeded 99% on both splits; minimum
held-out accuracy was 99.87%. It did not exhibit the strict near-chance
memorization plateau.

Width 32 first fitted training at epoch 575 but never reached joint 99%
within the cap. Its late generalization transition reached 96.41% held-out
accuracy at epoch 10,000. Width 16 remained below the training threshold.
Changing only weight decay to 0.2 brought width 32's transition inside the
budget. Training first reached 99% at epoch 825. A near-chance memorization
plateau began at epoch 1,050 and was confirmed at 1,150. Joint 99% accuracy
began at epoch 7,675 and satisfied the 20-observation criterion at 8,150.
The run finished at 100% on both splits; every observation in the final
1,000 epochs exceeded 99%, with minimum held-out accuracy 99.98%.
This 32,232-parameter model is the smallest tested delayed-grokking model
so far. Width 24 with the same stronger-decay recipe finished at 99.89%
training and 37.46% held-out accuracy, without a generalization event.
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
Named candidates run serially on the GPU:

```sh
uv run --locked scripts/run-mesh-search.py experiments/div97-mesh-search.json --dry-run
uv run --locked scripts/run-mesh-search.py experiments/div97-mesh-search.json \
  w512-h8-l2-d44-f44-g14-wd01
```

See the [portable results](grokking-results/summary.json) and
[learning curves](grokking-results/grokking-size-search.svg). The original
Mesh defaults are substantially
smaller than the user's successful parameter count, so their failure does
not settle the larger-model search.
