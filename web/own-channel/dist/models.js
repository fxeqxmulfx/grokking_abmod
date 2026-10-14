import { P, Random } from './model.js';
function dense(id, width) {
    return { id, title: 'Rotate90', subtitle: 'Dense + ReLU', width, seed: 11,
        learningRate: .001, weightDecay: .1,
        reportEvery: 10, parameters: P * width + width * width, embedding: P * width, gain: P * width };
}
export const MODELS = {
    'dense-rotary-16': dense('dense-rotary-16', 16),
    'dense-rotary-32': dense('dense-rotary-32', 32),
    'dense-rotary': dense('dense-rotary', 128),
    'dense-rotary-wide': dense('dense-rotary-wide', 1024),
};
export function modelId(id) {
    if (!Object.hasOwn(MODELS, id))
        throw new Error('Unknown model.');
    return id;
}
export function modelInfo(id) { return MODELS[id]; }
// Box–Muller uses the native SplitMix64 stream and rounds each stored value to
// f32. E consumes the same number of draws as the old uniform initializer,
// leaving every Kaiming-normal A parameter unchanged.
export function initialModelWeights(seed = 11, id = 'dense-rotary') {
    const { width, embedding, parameters } = modelInfo(id);
    const random = new Random(BigInt(seed) ^ 0x73696d706c650004n), weights = new Float32Array(parameters);
    for (const [start, end, variance] of [[0, embedding, 1 / width], [embedding, parameters, 2 / width]]) {
        const deviation = Math.sqrt(variance);
        for (let i = start; i < end; i += 2) {
            const radius = Math.sqrt(-2 * Math.log(1 - random.unit())), angle = 2 * Math.PI * random.unit();
            weights[i] = Math.fround(deviation * radius * Math.cos(angle));
            weights[i + 1] = Math.fround(deviation * radius * Math.sin(angle));
        }
    }
    return weights;
}
export function inputMatrix(weights, id = 'dense-rotary') {
    const W = modelInfo(id).width, result = new Float64Array(2 * P * W);
    result.set(weights.subarray(0, P * W));
    for (let k = 0; k < P; k++)
        for (let c = 0; c < W; c++)
            result[(P + k) * W + c] = (c % 2 ? 1 : -1) * weights[k * W + (c ^ 1)];
    return result;
}
export function readoutIndex(id, channel, answer) { return answer * modelInfo(id).width + channel; }
export function rms(values) {
    const inverse = 1 / Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length + 1e-5);
    return { inverse, values: values.map(value => value * inverse) };
}
export function predictModel(weights, a, b, id = 'dense-rotary', projected) {
    const { width: W, gain } = modelInfo(id), input = projected ?? inputMatrix(weights, id);
    const rawInput = Float64Array.from({ length: W }, (_, c) => input[a * W + c] + input[(P + b) * W + c]);
    const normalized = rms(rawInput), z = new Float64Array(W);
    for (let i = 0; i < W; i++)
        for (let c = 0; c < W; c++)
            z[c] += normalized.values[i] * weights[gain + i * W + c];
    const finalBefore = rawInput.map((value, c) => value + Math.max(0, z[c])), final = rms(finalBefore);
    const hidden = final.values, logits = new Float64Array(P);
    for (let c = 0; c < W; c++)
        for (let k = 0; k < P; k++)
            logits[k] += hidden[c] * weights[k * W + c];
    return { answer: logits.indexOf(Math.max(...logits)), hidden, logits, inputHidden: rawInput, rawInput,
        stages: [{ before: rawInput, normalized, values: [normalized.values, z], after: finalBefore }], finalBefore, final };
}
//# sourceMappingURL=models.js.map