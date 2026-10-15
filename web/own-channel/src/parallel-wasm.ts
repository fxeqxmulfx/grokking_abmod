import {cpuMemoryLayout} from './cpu-memory.ts';
import {browserCPUThread} from './cpu-protocol.ts';
import type {CPUThread, CPUThreadRequest} from './cpu-protocol.ts';
import {TOTAL, TRAIN} from './model.ts';
import {modelInfo} from './models.ts';
import type {TrainerOptions, Sample, Metrics, EngineInfo, EngineCheckpoint} from './types.ts';
import {WASMTrainer, validateSubmission} from './wasm.ts';

export interface ParallelOptions extends TrainerOptions {threads: number; workerFactory?: () => CPUThread}
interface Pending {job: number; resolve(): void; reject(error: Error): void}
interface Thread {port: CPUThread; pending: Pending | null}

export class ParallelWASMTrainer {
  static async create(seed = 11, options: ParallelOptions) {
    const layout = cpuMemoryLayout(modelInfo(options.model ?? 'dense-rotary').width, options.threads);
    const memory = new WebAssembly.Memory({initial: layout.pages, maximum: layout.pages, shared: true});
    const engine = await WASMTrainer.create(seed, {...options, shared: {memory, addresses: layout.slots[0]}});
    const trainer = new ParallelWASMTrainer(engine, layout, memory, options.threads);
    try {
      const ready: Promise<void>[] = [];
      for (let slot = 1; slot < options.threads; slot++) {
        const thread: Thread = {port: (options.workerFactory ?? browserCPUThread)(), pending: null};
        thread.port.onmessage = ({data}) => {
          if (data.type === 'error') {trainer.fail(new Error(data.message)); return;}
          if (thread.pending?.job !== data.job) {trainer.fail(new Error('Unexpected CPU worker response.')); return;}
          const pending = thread.pending; thread.pending = null; pending.resolve();
        };
        thread.port.onerror = event => trainer.fail(new Error(event.message || 'CPU worker failed.'));
        trainer.threads.push(thread);
        ready.push(trainer.send(thread, {type: 'initialize', module: engine.module, memory, addresses: layout.slots[slot], control: layout.control, slot}, 0));
      }
      await Promise.all(ready);
      return trainer;
    } catch (error) {trainer.destroy(); throw error;}
  }
  private threads: Thread[] = [];
  private control: Int32Array<ArrayBufferLike>;
  private job = 0; private busy = false; private failure: Error | null = null;
  constructor(readonly engine: WASMTrainer, readonly layout: ReturnType<typeof cpuMemoryLayout>, readonly memory: WebAssembly.Memory, readonly count: number) {
    this.control = new Int32Array(memory.buffer, layout.control, count * 16);
  }
  get step() {return this.engine.step;}
  private fail(error: Error) {
    this.failure ??= error;
    for (const thread of this.threads) if (thread.pending) {thread.pending.reject(error); thread.pending = null;}
  }
  private send(thread: Thread, request: CPUThreadRequest, job: number): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.failure) {reject(this.failure); return;}
      if (thread.pending) {reject(new Error('CPU worker already has a job.')); return;}
      thread.pending = {job, resolve, reject};
      try {thread.port.postMessage(request);} catch (error) {thread.pending = null; reject(error);}
    });
  }
  private begin() {
    if (this.failure) throw this.failure;
    if (this.busy) throw new Error('CPU trainer already has a job.');
    this.busy = true;
  }
  private phase() {const job = ++this.job; Atomics.store(this.control, 0, job); return job;}
  private acquire(slot: number, job: number) {
    if (Atomics.load(this.control, slot * 16) !== job) throw new Error('CPU worker did not finish its phase.');
  }
  async train(samples: Sample[], {gradientsOnly = false} = {}) {
    validateSubmission(samples, gradientsOnly); this.begin();
    try {
      for (const sample of samples) {
        this.engine.samples.set(sample);
        const active = Math.min(this.count, sample.length), job = this.phase();
        const stop = (slot: number) => Math.floor(sample.length * slot / active);
        const scale = Math.fround(1 / sample.length), pending: Promise<void>[] = [];
        for (let slot = 1; slot < active; slot++) pending.push(this.send(this.threads[slot - 1], {type: 'gradient', job, source: this.layout.slots[0].samples, first: stop(slot), count: stop(slot + 1) - stop(slot), scale}, job));
        this.engine.exports.gradient_batch_scaled(stop(1), scale);
        await Promise.all(pending);
        // One optimizer, one update; merge private gradients in a fixed order.
        for (let slot = 1; slot < active; slot++) {this.acquire(slot, job); this.engine.exports.add_gradient(this.layout.slots[slot].gradient);}
        if (!gradientsOnly) this.engine.applyGradient();
      }
      if (!gradientsOnly) await this.engine.yieldTask();
    } catch (error) {this.fail(error instanceof Error ? error : new Error(String(error))); throw error;}
    finally {this.busy = false;}
  }
  async evaluate(): Promise<Metrics> {
    this.begin();
    try {
      const job = this.phase(), stop = (slot: number) => Math.floor(TOTAL * slot / this.count);
      const pending = this.threads.map((thread, i) => this.send(thread, {type: 'evaluate', job, first: stop(i + 1), end: stop(i + 2)}, job));
      this.engine.exports.evaluate_range(0, stop(1));
      await Promise.all(pending);
      let trainLoss = 0, heldoutLoss = 0, trainCorrect = 0, heldoutCorrect = 0;
      for (let slot = 0; slot < this.count; slot++) {
        if (slot) this.acquire(slot, job);
        const address = this.layout.slots[slot].statistics;
        const loss = new Float64Array(this.memory.buffer, address, 2), correct = new Uint32Array(this.memory.buffer, address + 16, 2);
        trainLoss += loss[0]; heldoutLoss += loss[1]; trainCorrect += correct[0]; heldoutCorrect += correct[1];
      }
      const metrics = {step: this.step, epoch: this.step / 10, trainLoss: Math.fround(trainLoss / TRAIN), trainAccuracy: Math.fround(trainCorrect / TRAIN), heldoutLoss: Math.fround(heldoutLoss / TRAIN), heldoutAccuracy: Math.fround(heldoutCorrect / TRAIN)};
      if (![metrics.trainLoss, metrics.heldoutLoss, metrics.trainAccuracy, metrics.heldoutAccuracy].every(Number.isFinite)) throw new Error('Training diverged: metrics are not finite. Reset the experiment.');
      return metrics;
    } finally {this.busy = false;}
  }
  readWeights() {return this.engine.readWeights();}
  readGradients() {return this.engine.readGradients();}
  readLogits(ids?: number[]) {return this.engine.readLogits(ids);}
  info(): EngineInfo {return {...this.engine.info(), description: `${this.engine.info().description} · ${this.count} threads`, threads: this.count};}
  checkpoint() {if (this.busy) throw new Error('CPU trainer already has a job.'); return this.engine.checkpoint();}
  restore(checkpoint: EngineCheckpoint) {if (this.busy) throw new Error('CPU trainer already has a job.'); this.engine.restore(checkpoint);}
  destroy() {this.fail(new Error('CPU trainer stopped.')); for (const thread of this.threads) thread.port.terminate(); this.threads = []; this.engine.destroy();}
}
