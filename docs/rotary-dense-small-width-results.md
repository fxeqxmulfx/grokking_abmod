# Small-width Rotate90 Dense + ReLU results

Rust/CUDA, seed 11, NVIDIA GeForce RTX 3050 Laptop GPU, 2026-10-14.
W=32 and W=64 generalize to at least 99% test accuracy. W=16 does not learn
the training set within the 10,000-epoch budget with these hyperparameters.
These are individual seed results, not guarantees for other initializations.

| Width | Parameters | First train ≥99% epoch | First test ≥90% epoch | First test ≥99% epoch | Final train / test |
| --- | ---: | ---: | ---: | ---: | --- |
| 16 | 1,808 | Not reached | Not reached | Not reached | 32.09% / 18.11% (epoch 10,000) |
| 32 | 4,128 | 931 | 858 | 3690 | 99.98% / 99.03% |
| 64 | 10,304 | 460 | 477 | 560 | 100% / 99.03% |
| 128 (previous control) | 28,800 | 315 | 510 | 605 | See the normal-width initialization study |

W=64 reaches 99% test accuracy in 45 fewer epochs than W=128 (7.4% fewer).
This comparison concerns epochs, not browser throughput. W=64 had already
completed before the cancellation arrived; no additional W=64 runs were made.
The website adds W=16 and W=32 alongside its existing W=128 and W=1024 tabs.

The shared E is normal with variance 1/W; A is Kaiming normal with variance
2/W. The input is E[a] + Rotate90(E[b]). One RMSNorm → Dense A → ReLU branch
adds a residual skip; final RMSNorm and Eᵀ produce 97 logits. Normalization
has epsilon 1e-5 and no learned gain. All parameters use AdamW, lr=0.001,
weight decay=0.1, betas=(0.9,0.98), epsilon=1e-8 and a ten-update warmup.

The task is division modulo 97, with 4,656 train and 4,656 test equations.
Each epoch visits every training equation exactly once: nine batches of 512
and one batch of 48. Full train/test evaluation occurs after every epoch.
Runs stop at their first test ≥99% crossing or after 10,000 epochs.
W=16's best recorded train/test accuracies are 33.35% and 19.14%.

[Commands](../experiments/div97-dense-small-widths.args),
[metrics, run manifests and source/binary hashes](grokking-results/dense-small-widths/summary.json).
The reused binary's source hashes match the current native files and the
previous normal-width initialization study. The earlier W=128 control is
described in `rotary-dense-embedding-results.md`.
