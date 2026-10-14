# Browser introduction

The startup dialog explains late generalization, complete epochs, the original
paper, the learning curves, and the lab's numerical grokking criterion. The
learning curve's `?` button opens the shading explanation directly. Content is
available in English, Spanish, German, Japanese, French, and Russian.

A fixed budget such as 100 epochs is not a universal stopping rule. Memorizing
the training split can precede generalization by a long interval. Neither late
generalization nor a sharp transition is guaranteed for a particular width,
optimizer, or seed.

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

## Validation

The TypeScript build and seven targeted Node tests passed. An isolated Chrome
profile checked 360 layouts across six languages, four viewport sizes
(1366×768, 390×844, 320×568, 844×390), every introduction page, the model
explanations, and all main views. No page scrolling, clipped content, missing
SVGs, or JavaScript exceptions were observed. Language changes preserve live
training, and the chart guide opens its corresponding page.

Evidence: `web/own-channel/tests/results/education-2026-10-14/`.
