# Browser introduction

The startup dialog explains late generalization, complete epochs, the original
paper, the learning curves, the lab's numerical grokking criterion, and what
the lab's own runs have shown so far. The learning curve's `?` button opens the
shading explanation directly. Content is available in English, Spanish, German,
Japanese, French, and Russian.

The first page introduces the OpenAI study: small Transformers trained on
arithmetic tables such as division modulo 97 memorized the training examples
early and solved unseen pairs only much later, which the authors called
grokking. The lab presents itself as an attempt to find out in detail how this
works, by varying width, learning rate (lr), weight decay (wd) and seed while
following accuracy, loss and every weight. Neither late generalization nor a
sharp transition is guaranteed for a particular width, optimizer, or seed.

One browser epoch visits each of the 4,656 training equations once in shuffled
order: nine batches of 512 followed by one batch of 48, hence ten AdamW updates.
The 4,656 held-out equations do not contribute gradients. Accuracy measures
correct answers; cross-entropy measures the probability assigned to them.

## Paper and chart interpretation

[Power et al., 2022](https://arxiv.org/abs/2201.02177) studied small Transformers
on algorithmic datasets. The division-modulo-97 example reached nearly perfect
training accuracy before 1,000 optimizer steps, with validation approaching
that level around 1,000,000 steps. The local source is
`papers/arXiv-2201.02177v1/main.tex`. Its horizontal axis counts optimizer steps;
this browser counts complete epochs and uses a different, Dense architecture.

The chart's beige interval is the longest uninterrupted period with training
accuracy at least 99% and held-out accuracy at most 10%, before the first 90%
held-out crossing. Violet uses exactly 100% training accuracy under the same
held-out condition. Shading identifies memorization without generalization;
it says nothing about constant loss or constant weights.

The lab additionally requires a plateau lasting at least 25 epochs, a rise
from at most 10% to at least 90% held-out accuracy within 250 epochs, and a
99% crossing. These are reproducible lab thresholds, not the paper's universal
definition of grokking.

## Findings

The Grokking page and the Findings page summarize the Rust/CUDA
late-generalization grid in `docs/grokking-results/dense-optimizer-late/`:
lr 0.00003–1 × wd 0–1 for W32–W256 (lr 0.02–1 × wd 0–0.01 for W16), seeds 11,
17 and 23, at most 10,000 epochs, stopping at 99% held-out accuracy.

- W16–W64 never held a 25-epoch memorization plateau. W128 held one in 41 of
  165 runs, all at lr ≤ 0.001 and almost all with wd ≤ 0.01, but no run met the
  lab's grokking rule: plateau and rise lengthen together as 1/lr, so a rise
  within 250 epochs comes with a plateau shorter than 25. W256 held a plateau
  in 100 of 143 runs and met the rule in 19.
- At W128 with lr ≤ 0.003, the curves depend only on lr × epochs and wd.
  Halving lr roughly doubles every epoch count. Held-out accuracy reaches 99%
  at lr × epochs of 0.23–0.26 with wd 1, 0.40–0.45 with wd 0.3, 0.58–0.67 with
  wd 0.1 and 0.76–1.73 with wd ≤ 0.01. At lr 0.001 that is about 240, 400,
  600 and 860–1,290 epochs. The same bands predict which low-lr runs miss the
  10,000-epoch budget.
- Every setting with lr × wd ≥ 0.1 per step failed. lr 1 is erratic: only
  wd 0.0001 reached 99% on all three seeds. The fastest W128 setting, lr 0.01
  with wd 1, reaches 99% in 30–32 epochs; it is the site's Fast preset.

This agrees with the paper, where weight decay improved generalization the most
and the latest-generalizing example used Adam without weight decay. Batch size,
Adam betas and warmup were fixed here, so the grid says nothing about them.

## Validation

The TypeScript build and seven targeted Node tests passed. An isolated Chrome
profile checked 360 layouts across six languages, four viewport sizes
(1366×768, 390×844, 320×568, 844×390), every introduction page, the model
explanations, and all main views. No page scrolling, clipped content, missing
SVGs, or JavaScript exceptions were observed. Language changes preserve live
training, and the chart guide opens its corresponding page.

Evidence: `web/own-channel/tests/results/education-2026-10-14/`.

On 2026-10-15 the first page was rewritten and the Findings page added. The 84
Node tests passed, and the same Chrome check passed 402 layouts, including the
new page in every language and viewport. Below 380 px width the introduction
tabs use narrower side padding, and their labels never wrap.

Evidence: `web/own-channel/tests/results/education-2026-10-15/`.
