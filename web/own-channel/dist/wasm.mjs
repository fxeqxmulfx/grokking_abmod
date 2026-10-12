import {P, TOTAL, TRAIN, dataset} from './model.mjs';
import {modelInfo, initialModelWeights} from './models.mjs';

export class WASMTrainer {
  static async create(seed = 7, options = {}) {
    if (typeof WebAssembly !== 'object') throw new Error('This browser does not support WebAssembly.');
    let scalar = options.scalar ?? false;
    const model = options.model ?? 'original'; modelInfo(model);
    const load = async variant => {
      const response = await fetch(new URL(`./${model === 'original' ? 'train' : model}-${variant}.wasm`, import.meta.url));
      if (!response.ok) throw new Error(`Could not load WASM trainer: HTTP ${response.status}.`);
      return new Uint8Array(await response.arrayBuffer());
    };
    let bytes = options.bytes ?? await load(scalar ? 'scalar' : 'simd');
    if (!WebAssembly.validate(bytes)) {
      if (options.bytes || scalar) throw new Error('The WASM trainer is not supported by this browser.');
      scalar = true; bytes = await load('scalar');
    }
    const {instance} = await WebAssembly.instantiate(bytes, {});
    return new WASMTrainer(instance, seed, {...options, scalar, model});
  }
  constructor(instance, seed, options) {
    this.instance = instance; this.exports = instance.exports; this.scalar = options.scalar;
    this.model = modelInfo(options.model); const parameters = this.model.parameters;
    this.step = 0; this.b1 = 1; this.b2 = 1;
    this.learningRate = options.learningRate ?? .001; this.decay = options.decay ?? .1;
    const view = (name, type, length) => new type(this.exports.memory.buffer, this.exports[name].value, length);
    this.weights = view('weights', Float32Array, parameters);
    this.gradients = view('gradient', Float32Array, parameters);
    this.samples = view('samples', Uint32Array, 512);
    this.metrics = view('metrics', Float32Array, 4);
    this.logits = view('logits', Float32Array, 100);
    this.data = dataset(seed); view('data', Uint32Array, TOTAL * 3).set(this.data);
    const weights = options.weights ?? initialModelWeights(seed, this.model.id);
    if (weights.length !== parameters || !weights.every(Number.isFinite)) throw new Error('Invalid model weights.');
    this.weights.set(weights); this.exports.prepare();
    // A resolved Promise only yields microtasks. MessageChannel yields a task
    // so the worker receives Pause/Reset even during CPU-only training.
    this.channel = new MessageChannel();
    this.yieldTask = () => new Promise(resolve => { this.channel.port1.onmessage = resolve; this.channel.port2.postMessage(null); });
  }
  async train(samples, {gradientsOnly = false} = {}) {
    if (!samples.length || samples.length > 32 || (gradientsOnly && samples.length !== 1)) throw new Error('A submission needs 1..32 minibatches, or one gradient batch.');
    for (const sample of samples) {
      if (!sample.length || sample.length > 512 || !sample.every(id => Number.isInteger(id) && id >= 0 && id < TRAIN)) throw new Error('Invalid minibatch.');
      this.samples.set(sample);
      if (gradientsOnly) this.exports.gradient_batch(sample.length);
      else {
        this.step++; this.b1 = Math.fround(this.b1 * Math.fround(.9)); this.b2 = Math.fround(this.b2 * Math.fround(.98));
        const rate = Math.fround(Math.fround(this.learningRate) * Math.fround(Math.min(this.step / 10, 1)));
        this.exports.step(sample.length, rate, this.decay, Math.fround(1 - this.b1), Math.fround(1 - this.b2));
      }
    }
    if (!gradientsOnly) await this.yieldTask();
  }
  async evaluate() {
    this.exports.evaluate();
    if (!this.metrics.every(Number.isFinite)) throw new Error('Training diverged: metrics are not finite. Reset the experiment.');
    return {step: this.step, epoch: this.step / 10, trainLoss: this.metrics[0], trainAccuracy: this.metrics[1], heldoutLoss: this.metrics[2], heldoutAccuracy: this.metrics[3]};
  }
  async readWeights() { return this.weights.slice(); }
  async readGradients() { return this.gradients.slice(); }
  async readLogits(ids = [0, 19, 4655, 4656, 9311]) {
    return ids.map(id => { this.exports.forward(id); return this.logits.slice(0, P); });
  }
  info() { return {description: this.scalar ? 'CPU · WASM scalar' : 'CPU · WASM SIMD128', backend: 'wasm', fallback: false, memoryBytes: this.exports.memory.buffer.byteLength}; }
  destroy() { this.channel.port1.close(); this.channel.port2.close(); this.instance = null; this.exports = null; }
}
