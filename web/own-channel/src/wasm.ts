import type {Weights, Sample, Metrics, TrainerOptions, EngineInfo} from './types.ts';

export interface TrainerExports {
  memory: WebAssembly.Memory; weights: WebAssembly.Global; gradient: WebAssembly.Global;
  samples: WebAssembly.Global; metrics: WebAssembly.Global; logits: WebAssembly.Global; data: WebAssembly.Global;
  statistics: WebAssembly.Global;
  prepare(): void; gradient_batch(rows: number): void; gradient_batch_scaled(rows: number, scale: number): void;
  add_gradient(source: number): void;
  apply_update(rate: number, decay: number, correction1: number, correction2: number): void;
  step(rows: number, rate: number, decay: number, correction1: number, correction2: number): void;
  evaluate(): void; evaluate_range(first: number, end: number): void; forward(id: number): void;
}
import {P, TOTAL, TRAIN, dataset} from './model.ts';
import {modelInfo, initialModelWeights} from './models.ts';

export class WASMTrainer {
  static async create(seed = 11, options: TrainerOptions = {}) {
    if (typeof WebAssembly !== 'object') throw new Error('This browser does not support WebAssembly.');
    let scalar = options.scalar ?? false;
    const model = options.model ?? 'dense-rotary'; modelInfo(model);
    const load = async (variant: 'scalar' | 'simd') => {
      const response = await fetch(new URL(`./${model}-${variant}${options.shared ? '-shared' : ''}.wasm`, import.meta.url));
      if (!response.ok) throw new Error(`Could not load WASM trainer: HTTP ${response.status}.`);
      return new Uint8Array(await response.arrayBuffer());
    };
    let bytes = options.bytes ?? await load(scalar ? 'scalar' : 'simd');
    if (!WebAssembly.validate(bytes)) {
      if (options.bytes || scalar || options.requireSIMD) throw new Error('The WASM trainer is not supported by this browser.');
      scalar = true; bytes = await load('scalar');
    }
    const {instance, module} = await WebAssembly.instantiate(bytes, options.shared ? {env: {memory: options.shared.memory, ...options.shared.addresses}} : {});
    const trainer = new WASMTrainer(instance, seed, {...options, scalar, model});
    trainer.module = module;
    return trainer;
  }
  instance: WebAssembly.Instance; exports: TrainerExports; scalar: boolean; model: ReturnType<typeof modelInfo>;
  module!: WebAssembly.Module;
  weights: Float32Array<ArrayBufferLike>; gradients: Float32Array<ArrayBufferLike>; samples: Uint32Array<ArrayBufferLike>; metrics: Float32Array<ArrayBufferLike>; logits: Float32Array<ArrayBufferLike>;
  data: Uint32Array<ArrayBuffer>; step = 0; b1 = 1; b2 = 1; learningRate: number; decay: number;
  channel: MessageChannel; yieldTask: () => Promise<void>;
  constructor(instance: WebAssembly.Instance, seed: number, options: TrainerOptions & {scalar: boolean; model: NonNullable<TrainerOptions['model']>}) {
    this.instance = instance; this.exports = instance.exports as unknown as TrainerExports; this.scalar = options.scalar;
    this.model = modelInfo(options.model); const parameters = this.model.parameters;
    this.step = 0; this.b1 = 1; this.b2 = 1;
    this.learningRate = options.learningRate ?? .001; this.decay = options.decay ?? .1;
    for (const name of ['prepare', 'gradient_batch', 'step', 'evaluate', 'forward'] as const) if (typeof this.exports[name] !== 'function') throw new Error('Invalid WASM trainer exports.');
    if (!(this.exports.memory instanceof WebAssembly.Memory)) throw new Error('Invalid WASM trainer memory.');
    type Address = 'weights' | 'gradient' | 'samples' | 'metrics' | 'logits' | 'data';
    const address = (name: Address) => {
      if (!(this.exports[name] instanceof WebAssembly.Global) || !Number.isInteger(this.exports[name].value)) throw new Error('Invalid WASM trainer address.');
      return Number(this.exports[name].value);
    };
    const floats = (name: Address, length: number) => new Float32Array(this.exports.memory.buffer, address(name), length);
    const integers = (name: Address, length: number) => new Uint32Array(this.exports.memory.buffer, address(name), length);
    this.weights = floats('weights', parameters);
    this.gradients = floats('gradient', parameters);
    this.samples = integers('samples', 512);
    this.metrics = floats('metrics', 4);
    this.logits = floats('logits', 100);
    this.data = dataset(seed); integers('data', TOTAL * 3).set(this.data);
    const weights = options.weights ?? initialModelWeights(seed, this.model.id);
    if (weights.length !== parameters || !weights.every(Number.isFinite)) throw new Error('Invalid model weights.');
    this.weights.set(weights); this.exports.prepare();
    // A resolved Promise only yields microtasks. MessageChannel yields a task
    // so the worker receives Pause/Reset even during CPU-only training.
    this.channel = new MessageChannel();
    this.yieldTask = () => new Promise(resolve => { this.channel.port1.onmessage = () => resolve(); this.channel.port2.postMessage(null); });
  }
  async train(samples: Sample[], {gradientsOnly = false} = {}) {
    validateSubmission(samples, gradientsOnly);
    for (const sample of samples) {
      this.samples.set(sample);
      this.exports.gradient_batch(sample.length);
      if (!gradientsOnly) this.applyGradient();
    }
    if (!gradientsOnly) await this.yieldTask();
  }
  applyGradient() {
    this.step++; this.b1 = Math.fround(this.b1 * Math.fround(.9)); this.b2 = Math.fround(this.b2 * Math.fround(.98));
    const rate = Math.fround(Math.fround(this.learningRate) * Math.fround(Math.min(this.step / 10, 1)));
    this.exports.apply_update(rate, this.decay, Math.fround(1 - this.b1), Math.fround(1 - this.b2));
  }
  async evaluate(): Promise<Metrics> {
    this.exports.evaluate();
    if (!this.metrics.every(Number.isFinite)) throw new Error('Training diverged: metrics are not finite. Reset the experiment.');
    return {step: this.step, epoch: this.step / 10, trainLoss: this.metrics[0], trainAccuracy: this.metrics[1], heldoutLoss: this.metrics[2], heldoutAccuracy: this.metrics[3]};
  }
  async readWeights() { return this.weights.slice(); }
  async readGradients() { return this.gradients.slice(); }
  async readLogits(ids = [0, 19, 4655, 4656, 9311]) {
    return ids.map(id => { this.exports.forward(id); return this.logits.slice(0, P); });
  }
  info(): EngineInfo { return {description: this.scalar ? 'CPU · WASM scalar' : 'CPU · WASM SIMD128', backend: 'wasm', threads: 1, memoryBytes: this.exports.memory.buffer.byteLength}; }
  destroy() { this.channel.port1.close(); this.channel.port2.close(); }
}
export function validateSubmission(samples: Sample[], gradientsOnly = false): void {
  if (!samples.length || samples.length > 32 || (gradientsOnly && samples.length !== 1)) throw new Error('A submission needs 1..32 minibatches, or one gradient batch.');
  for (const sample of samples) if (!sample.length || sample.length > 512 || !sample.every(id => Number.isInteger(id) && id >= 0 && id < TRAIN)) throw new Error('Invalid minibatch.');
}
