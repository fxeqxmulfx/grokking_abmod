export type GuidePage = 'epoch' | 'chart' | 'weights' | 'probe' | 'grokking' | 'findings' | 'task' | 'model';
export interface HelpTopic {title: string; paragraphs: readonly string[]; formula?: string; guide?: GuidePage}

export const HELP_TOPICS = {
  width: {title: 'Width', paragraphs: [
    'W is the number of channels: the values the model uses to represent each input. Choose 16, 32, 64, 128 or 256. A wider model has more weights and needs more time and memory.',
    'Each width and preset keeps its own experiment while this page is open. Switching back restores its weights and learning curves.',
  ], guide: 'model'},
  presets: {title: 'Optimizer preset', paragraphs: [
    'Fast, Plateau and Slow use the same architecture with different learning rates and weight decay. Fast learned unseen pairs earliest in the reference runs; Slow learned them latest. Plateau was selected for a long period of memorization.',
    'These names describe the reference experiments, not a guaranteed result in your browser. Use Settings to choose your own lr and wd.',
  ], guide: 'findings'},
  settings: {title: 'Custom experiment', paragraphs: [
    'Choose a width, learning rate and weight decay, then apply them. This starts a new Custom experiment from the selected seed. Applying again replaces that Custom run; the built-in presets keep their results.',
  ]},
  lr: {title: 'Learning rate (lr)', paragraphs: [
    'The learning rate controls the size of each weight update. Smaller values usually need more epochs. Values that are too large can make training unstable. Start with the preset value and change one setting at a time.',
  ], guide: 'findings'},
  wd: {title: 'Weight decay (wd)', paragraphs: [
    'Weight decay gently shrinks weights during AdamW training. It can help the model learn a rule instead of only memorizing examples. Zero turns this shrinkage off; a larger value does not always give a better result.',
  ], guide: 'findings'},
  curves: {title: 'Learning curves', paragraphs: [
    'Orange is training; blue is held-out test data. Compare the curves to see whether the model learns unseen pairs.',
    'Loss measures prediction error; lower is better. Accuracy is the fraction of correct answers; higher is better.',
  ], guide: 'chart'},
  'train-accuracy': {title: 'Train accuracy', paragraphs: [
    'The percentage of the 4,656 training pairs answered correctly. A high value can mean the model has memorized these pairs. Compare it with held-out accuracy to see whether the model also solves unseen pairs.',
  ], guide: 'chart'},
  'test-accuracy': {title: 'Held-out accuracy', paragraphs: [
    'The percentage of the 4,656 test pairs answered correctly. These pairs never update the weights, so this measures how well the model handles unseen examples. Training pauses once this value reaches 99%; you can continue afterward.',
  ], guide: 'chart'},
  'train-loss': {title: 'Train loss', paragraphs: [
    'The average cross-entropy error on the training pairs. It measures how much probability the model gives the correct answer. Lower is better; a confident wrong answer raises the loss.',
  ], guide: 'chart'},
  'test-loss': {title: 'Held-out loss', paragraphs: [
    'The same cross-entropy error, measured on test pairs that do not update the weights. It can improve before accuracy changes because the model can give the correct answer more probability without yet choosing it.',
  ], guide: 'chart'},
  training: {title: 'Training desk', paragraphs: [
    'Start or continue training with the main button. Pause keeps the weights and optimizer state. +1 epoch trains one complete pass; Reset starts this experiment again from its seed.',
    'Experiments are kept in this open page, not saved permanently. Reloading or closing the page clears them. Export metrics if you want to keep the learning curves.',
  ]},
  epoch: {title: 'Epoch', paragraphs: [
    'One epoch uses all 4,656 training equations once, in shuffled order.',
    'One optimizer step updates weights from one batch. Each epoch contains ten steps.',
    'Test pairs are evaluated after every epoch and never update weights.',
  ], guide: 'epoch'},
  seed: {title: 'Seed', paragraphs: [
    'The seed sets the initial weights and the order of training examples. Keep it fixed to compare settings fairly. Changing it restarts the current experiment; the same seed does not guarantee identical results across different training engines.',
  ]},
  limit: {title: 'Epoch limit', paragraphs: [
    'The maximum number of epochs for this run, from 1 to 10,000. Reaching the limit pauses training. Increase the limit and continue to train longer; reducing it does not undo completed epochs.',
  ], guide: 'epoch'},
  engine: {title: 'Training engine', paragraphs: [
    'Automatic chooses a CPU engine for this browser. SIMD calculates several values at once; Scalar calculates them one at a time. More threads can help, but also use more memory and CPU capacity.',
    'Changing the engine restarts the current experiment. On phones, Automatic uses at most two threads to leave room for the interface.',
  ]},
  capture: {title: 'Weight updates', paragraphs: [
    'How often the displayed weights refresh: 1, 5 or 10 times per second. This changes the display rate, not the learning rate or the number of training steps. Accuracy and loss are still measured after every epoch.',
  ]},
  performance: {title: 'Training speed', paragraphs: [
    'Updates per second counts optimizer steps, not epochs. There are ten steps per epoch. The timer counts active training time and excludes pauses and engine preparation.',
  ]},
  phase: {title: 'Learning phase', paragraphs: [
    'This label summarizes the accuracy history. A memorization plateau means training accuracy is high while test accuracy stays low. Generalizing means the model solves some unseen pairs; the grokking label requires the full plateau-and-rise criterion.',
  ], guide: 'grokking'},
  exports: {title: 'Export results', paragraphs: [
    'Export metrics downloads a CSV file with loss and accuracy for every recorded epoch. Open it in a spreadsheet or use it to make your own plots.',
    'PyTorch .py downloads code for this architecture, seed and settings. It starts a fresh training run; it does not export the currently trained weights.',
  ]},
  network: {title: 'Network', paragraphs: [
    'The top row contains the two inputs; the bottom row contains the 97 possible answers. The middle rows are the model’s channels. Each line is a weight: pink is positive and blue is negative.',
    'All shows every connection. E is reused for both inputs and the output; A connects the middle rows. Channel follows one output channel through the Dense layer, ReLU and the residual connection.',
  ], guide: 'model'},
  'weight-view': {title: 'Weights and Change ×4', paragraphs: [
    'Weights shows current values. Change ×4 shows the current weight minus its initial value, with a four-times more sensitive color scale.',
    'In Change ×4, pink marks an increase and blue a decrease. A neutral cell has no change.',
  ], guide: 'weights'},
  highlight: {title: 'Highlight', paragraphs: [
    'Emphasize the selected channel’s connections without hiding the other weights. Choose a channel with the Channel field or a contribution bar. Turning this off returns to an overview of the whole network.',
  ]},
  channel: {title: 'Channel', paragraphs: [
    'A channel is one component of the model’s internal representation. Indices start at 0 and end at W − 1. Selecting one links the network view, the marked columns on the weight maps and the contribution bars.',
  ], guide: 'probe'},
  zoom: {title: 'Zoom and pan', paragraphs: [
    'Use + and − to spread the channels apart or bring them closer. Drag the network to move it; arrow keys also move the focused graph. Fit restores the full view. These controls only change the display.',
  ]},
  weights: {title: 'Weight atlas', paragraphs: [
    'Each cell is one weight. Pink is positive, blue is negative, and a darker color means a larger magnitude. The color scale stays fixed during training so you can compare different moments.',
    'E and the output readout share one set of weights. The three maps show different roles, not three independent matrices. Tap a column to select its channel.',
  ], guide: 'weights'},
  embedding: {title: 'Shared embedding E', paragraphs: [
    'E turns each number from 0 to 96 into a row of W channel values. Rows are numbers; columns are channels. The same matrix is used for both operands and, transposed, for scoring the answers.',
  ], guide: 'weights'},
  dense: {title: 'Dense matrix A', paragraphs: [
    'A connects every input channel to every output channel in the Dense layer. Rows are input channels; columns are output channels. Each cell is one learned connection, so this matrix contains W × W weights.',
  ], guide: 'weights'},
  readout: {title: 'Tied readout Eᵀ', paragraphs: [
    'The output readout turns the final channel values into scores for the 97 possible answers. R names these output weights; here R = Eᵀ. The map places answers in rows and channels in columns.',
  ], guide: 'weights'},
  centered: {title: 'What does Center R mean?', paragraphs: [
    "R means readout: the model's output weights, here Eᵀ.",
    'For each channel, Center R subtracts the mean weight across all 97 answers. Colors show each answer’s deviation from that channel’s average weight.',
    'This changes only the readout heatmap. Weights, training and predictions stay the same.',
  ], formula: 'R[c,q] − mean_q R[c,q]', guide: 'weights'},
  probe: {title: 'Probe the model', paragraphs: [
    'Try a division problem with a from 0 to 96 and b from 1 to 96. The model predicts q such that b × q leaves remainder a when divided by 97. Changing these inputs does not train the model.',
    'Confidence is the probability assigned to the predicted answer for this example. A confident prediction can still be wrong. The contribution bars describe this same pair of inputs.',
  ], guide: 'task'},
  contributions: {title: 'Probe contributions', paragraphs: [
    'Each bar measures the size of one channel’s contribution to the current probe. A taller bar means a larger score, not necessarily more support for the correct answer.',
    'Top 16 is the share of the total score carried by the 16 strongest channels. It is not prediction confidence. Tap a bar to select its channel.',
  ], guide: 'probe'},
} satisfies Record<string, HelpTopic>;

export type HelpId = keyof typeof HELP_TOPICS;
