import { P, TOTAL, TRAIN, dataset } from './model.js';
import { modelInfo, initialModelWeights } from './models.js';
export class WASMTrainer {
    static async create(seed = 11, options = {}) {
        if (typeof WebAssembly !== 'object')
            throw new Error('This browser does not support WebAssembly.');
        let scalar = options.scalar ?? false;
        const model = options.model ?? 'dense-rotary';
        modelInfo(model);
        const load = async (variant) => {
            const response = await fetch(new URL(`./${model}-${variant}${options.shared ? '-shared' : ''}.wasm`, import.meta.url));
            if (!response.ok)
                throw new Error(`Could not load WASM trainer: HTTP ${response.status}.`);
            return new Uint8Array(await response.arrayBuffer());
        };
        let bytes = options.bytes ?? await load(scalar ? 'scalar' : 'simd');
        if (!WebAssembly.validate(bytes)) {
            if (options.bytes || scalar || options.requireSIMD)
                throw new Error('The WASM trainer is not supported by this browser.');
            scalar = true;
            bytes = await load('scalar');
        }
        const { instance, module } = await WebAssembly.instantiate(bytes, options.shared ? { env: { memory: options.shared.memory, ...options.shared.addresses } } : {});
        const trainer = new WASMTrainer(instance, seed, { ...options, scalar, model });
        trainer.module = module;
        return trainer;
    }
    instance;
    exports;
    scalar;
    model;
    module;
    weights;
    gradients;
    samples;
    metrics;
    logits;
    data;
    step = 0;
    b1 = 1;
    b2 = 1;
    learningRate;
    decay;
    channel;
    yieldTask;
    constructor(instance, seed, options) {
        this.instance = instance;
        this.exports = instance.exports;
        this.scalar = options.scalar;
        this.model = modelInfo(options.model);
        const parameters = this.model.parameters;
        this.step = 0;
        this.b1 = 1;
        this.b2 = 1;
        this.learningRate = options.learningRate ?? this.model.learningRate;
        this.decay = options.decay ?? this.model.weightDecay;
        for (const name of ['prepare', 'gradient_batch', 'step', 'evaluate', 'forward'])
            if (typeof this.exports[name] !== 'function')
                throw new Error('Invalid WASM trainer exports.');
        if (!(this.exports.memory instanceof WebAssembly.Memory))
            throw new Error('Invalid WASM trainer memory.');
        const address = (name) => {
            if (!(this.exports[name] instanceof WebAssembly.Global) || !Number.isInteger(this.exports[name].value))
                throw new Error('Invalid WASM trainer address.');
            return Number(this.exports[name].value);
        };
        const floats = (name, length) => new Float32Array(this.exports.memory.buffer, address(name), length);
        const integers = (name, length) => new Uint32Array(this.exports.memory.buffer, address(name), length);
        this.weights = floats('weights', parameters);
        this.gradients = floats('gradient', parameters);
        this.samples = integers('samples', 512);
        this.metrics = floats('metrics', 4);
        this.logits = floats('logits', 100);
        this.data = dataset(seed);
        integers('data', TOTAL * 3).set(this.data);
        const weights = options.weights ?? initialModelWeights(seed, this.model.id);
        if (weights.length !== parameters || !weights.every(Number.isFinite))
            throw new Error('Invalid model weights.');
        this.weights.set(weights);
        this.exports.prepare();
        // A resolved Promise only yields microtasks. MessageChannel yields a task
        // so the worker receives Pause/Reset even during CPU-only training.
        this.channel = new MessageChannel();
        this.yieldTask = () => new Promise(resolve => { this.channel.port1.onmessage = () => resolve(); this.channel.port2.postMessage(null); });
    }
    async train(samples, { gradientsOnly = false } = {}) {
        validateSubmission(samples, gradientsOnly);
        for (const sample of samples) {
            this.samples.set(sample);
            this.exports.gradient_batch(sample.length);
            if (!gradientsOnly)
                this.applyGradient();
        }
        if (!gradientsOnly)
            await this.yieldTask();
    }
    applyGradient() {
        this.step++;
        this.b1 = Math.fround(this.b1 * Math.fround(.9));
        this.b2 = Math.fround(this.b2 * Math.fround(.98));
        const rate = Math.fround(Math.fround(this.learningRate) * Math.fround(Math.min(this.step / 10, 1)));
        this.exports.apply_update(rate, this.decay, Math.fround(1 - this.b1), Math.fround(1 - this.b2));
    }
    async evaluate() {
        this.exports.evaluate();
        if (!this.metrics.every(Number.isFinite))
            throw new Error('Training diverged: metrics are not finite. Reset the experiment.');
        return { step: this.step, epoch: this.step / 10, trainLoss: this.metrics[0], trainAccuracy: this.metrics[1], heldoutLoss: this.metrics[2], heldoutAccuracy: this.metrics[3] };
    }
    async readWeights() { return this.weights.slice(); }
    async readGradients() { return this.gradients.slice(); }
    async readLogits(ids = [0, 19, 4655, 4656, 9311]) {
        return ids.map(id => { this.exports.forward(id); return this.logits.slice(0, P); });
    }
    info() { return { description: this.scalar ? 'CPU · WASM scalar' : 'CPU · WASM SIMD128', backend: 'wasm', threads: 1, memoryBytes: this.exports.memory.buffer.byteLength }; }
    checkpoint() {
        const moment = (name) => new Float32Array(this.exports.memory.buffer, Number(this.exports[name].value), this.weights.length).slice();
        return { model: this.model.id, step: this.step, b1: this.b1, b2: this.b2, weights: this.weights.slice(), moment1: moment('moment1'), moment2: moment('moment2') };
    }
    restore(checkpoint) {
        if (checkpoint.model !== this.model.id || !Number.isSafeInteger(checkpoint.step) || checkpoint.step < 0 || ![checkpoint.b1, checkpoint.b2].every(value => Number.isFinite(value) && value >= 0 && value <= 1)
            || ![checkpoint.weights, checkpoint.moment1, checkpoint.moment2].every(values => values.length === this.weights.length && values.every(Number.isFinite)))
            throw new Error('Invalid training checkpoint.');
        this.weights.set(checkpoint.weights);
        for (const name of ['moment1', 'moment2'])
            new Float32Array(this.exports.memory.buffer, Number(this.exports[name].value), this.weights.length).set(checkpoint[name]);
        this.step = checkpoint.step;
        this.b1 = checkpoint.b1;
        this.b2 = checkpoint.b2;
        this.exports.prepare();
    }
    destroy() { this.channel.port1.close(); this.channel.port2.close(); }
}
export function validateSubmission(samples, gradientsOnly = false) {
    if (!samples.length || samples.length > 32 || (gradientsOnly && samples.length !== 1))
        throw new Error('A submission needs 1..32 minibatches, or one gradient batch.');
    for (const sample of samples)
        if (!sample.length || sample.length > 512 || !sample.every(id => Number.isInteger(id) && id >= 0 && id < TRAIN))
            throw new Error('Invalid minibatch.');
}
//# sourceMappingURL=wasm.js.map