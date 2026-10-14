import { modelInfo } from './models.js';
import { PYTORCH_TEMPLATE } from './pytorch-template.js';
// Export source with the selected experiment defaults. Trained weights and
// optimizer state are never part of this download.
export function pytorchSource(options) {
    if (!Number.isInteger(options.seed) || options.seed < 0 || options.seed > 0xffffffff)
        throw new Error('Invalid export seed.');
    if (!Number.isInteger(options.maxEpochs) || options.maxEpochs < 1 || options.maxEpochs > 10000)
        throw new Error('Invalid export epoch limit.');
    return PYTORCH_TEMPLATE
        .replace(/^DEFAULT_WIDTH = \d+$/m, `DEFAULT_WIDTH = ${modelInfo(options.model).width}`)
        .replace(/^DEFAULT_LEARNING_RATE = [\d.]+$/m, `DEFAULT_LEARNING_RATE = ${modelInfo(options.model).learningRate}`)
        .replace(/^DEFAULT_WEIGHT_DECAY = [\d.]+$/m, `DEFAULT_WEIGHT_DECAY = ${modelInfo(options.model).weightDecay}`)
        .replace(/^DEFAULT_SEED = \d+$/m, `DEFAULT_SEED = ${options.seed}`)
        .replace(/^DEFAULT_EPOCHS = \d+$/m, `DEFAULT_EPOCHS = ${options.maxEpochs}`);
}
//# sourceMappingURL=pytorch-export.js.map