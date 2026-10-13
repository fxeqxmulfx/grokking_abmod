import { cpuMemoryLayout } from './cpu-memory.js';
import { browserCPUThread } from './cpu-protocol.js';
import { TOTAL, TRAIN } from './model.js';
import { modelInfo } from './models.js';
import { WASMTrainer, validateSubmission } from './wasm.js';
export class ParallelWASMTrainer {
    engine;
    layout;
    memory;
    count;
    static async create(seed = 11, options) {
        const layout = cpuMemoryLayout(modelInfo(options.model ?? 'dense-rotary').width, options.threads);
        const memory = new WebAssembly.Memory({ initial: layout.pages, maximum: layout.pages, shared: true });
        const engine = await WASMTrainer.create(seed, { ...options, shared: { memory, addresses: layout.slots[0] } });
        const trainer = new ParallelWASMTrainer(engine, layout, memory, options.threads);
        try {
            const ready = [];
            for (let slot = 1; slot < options.threads; slot++) {
                const thread = { port: (options.workerFactory ?? browserCPUThread)(), pending: null };
                thread.port.onmessage = ({ data }) => {
                    if (data.type === 'error') {
                        trainer.fail(new Error(data.message));
                        return;
                    }
                    if (thread.pending?.job !== data.job) {
                        trainer.fail(new Error('Unexpected CPU worker response.'));
                        return;
                    }
                    const pending = thread.pending;
                    thread.pending = null;
                    pending.resolve();
                };
                thread.port.onerror = event => trainer.fail(new Error(event.message || 'CPU worker failed.'));
                trainer.threads.push(thread);
                ready.push(trainer.send(thread, { type: 'initialize', module: engine.module, memory, addresses: layout.slots[slot], control: layout.control, slot }, 0));
            }
            await Promise.all(ready);
            return trainer;
        }
        catch (error) {
            trainer.destroy();
            throw error;
        }
    }
    threads = [];
    control;
    job = 0;
    busy = false;
    failure = null;
    constructor(engine, layout, memory, count) {
        this.engine = engine;
        this.layout = layout;
        this.memory = memory;
        this.count = count;
        this.control = new Int32Array(memory.buffer, layout.control, count * 16);
    }
    get step() { return this.engine.step; }
    fail(error) {
        this.failure ??= error;
        for (const thread of this.threads)
            if (thread.pending) {
                thread.pending.reject(error);
                thread.pending = null;
            }
    }
    send(thread, request, job) {
        return new Promise((resolve, reject) => {
            if (this.failure) {
                reject(this.failure);
                return;
            }
            if (thread.pending) {
                reject(new Error('CPU worker already has a job.'));
                return;
            }
            thread.pending = { job, resolve, reject };
            try {
                thread.port.postMessage(request);
            }
            catch (error) {
                thread.pending = null;
                reject(error);
            }
        });
    }
    begin() {
        if (this.failure)
            throw this.failure;
        if (this.busy)
            throw new Error('CPU trainer already has a job.');
        this.busy = true;
    }
    phase() { const job = ++this.job; Atomics.store(this.control, 0, job); return job; }
    acquire(slot, job) {
        if (Atomics.load(this.control, slot * 16) !== job)
            throw new Error('CPU worker did not finish its phase.');
    }
    async train(samples, { gradientsOnly = false } = {}) {
        validateSubmission(samples, gradientsOnly);
        this.begin();
        try {
            for (const sample of samples) {
                this.engine.samples.set(sample);
                const active = Math.min(this.count, sample.length), job = this.phase();
                const stop = (slot) => Math.floor(sample.length * slot / active);
                const scale = Math.fround(1 / sample.length), pending = [];
                for (let slot = 1; slot < active; slot++)
                    pending.push(this.send(this.threads[slot - 1], { type: 'gradient', job, source: this.layout.slots[0].samples, first: stop(slot), count: stop(slot + 1) - stop(slot), scale }, job));
                this.engine.exports.gradient_batch_scaled(stop(1), scale);
                await Promise.all(pending);
                // One optimizer, one update; merge private gradients in a fixed order.
                for (let slot = 1; slot < active; slot++) {
                    this.acquire(slot, job);
                    this.engine.exports.add_gradient(this.layout.slots[slot].gradient);
                }
                if (!gradientsOnly)
                    this.engine.applyGradient();
            }
            if (!gradientsOnly)
                await this.engine.yieldTask();
        }
        catch (error) {
            this.fail(error instanceof Error ? error : new Error(String(error)));
            throw error;
        }
        finally {
            this.busy = false;
        }
    }
    async evaluate() {
        this.begin();
        try {
            const job = this.phase(), stop = (slot) => Math.floor(TOTAL * slot / this.count);
            const pending = this.threads.map((thread, i) => this.send(thread, { type: 'evaluate', job, first: stop(i + 1), end: stop(i + 2) }, job));
            this.engine.exports.evaluate_range(0, stop(1));
            await Promise.all(pending);
            let trainLoss = 0, heldoutLoss = 0, trainCorrect = 0, heldoutCorrect = 0;
            for (let slot = 0; slot < this.count; slot++) {
                if (slot)
                    this.acquire(slot, job);
                const address = this.layout.slots[slot].statistics;
                const loss = new Float64Array(this.memory.buffer, address, 2), correct = new Uint32Array(this.memory.buffer, address + 16, 2);
                trainLoss += loss[0];
                heldoutLoss += loss[1];
                trainCorrect += correct[0];
                heldoutCorrect += correct[1];
            }
            const metrics = { step: this.step, epoch: this.step / 10, trainLoss: Math.fround(trainLoss / TRAIN), trainAccuracy: Math.fround(trainCorrect / TRAIN), heldoutLoss: Math.fround(heldoutLoss / TRAIN), heldoutAccuracy: Math.fround(heldoutCorrect / TRAIN) };
            if (![metrics.trainLoss, metrics.heldoutLoss, metrics.trainAccuracy, metrics.heldoutAccuracy].every(Number.isFinite))
                throw new Error('Training diverged: metrics are not finite. Reset the experiment.');
            return metrics;
        }
        finally {
            this.busy = false;
        }
    }
    readWeights() { return this.engine.readWeights(); }
    readGradients() { return this.engine.readGradients(); }
    readLogits(ids) { return this.engine.readLogits(ids); }
    info() { return { ...this.engine.info(), description: `${this.engine.info().description} · ${this.count} threads`, threads: this.count }; }
    destroy() { this.fail(new Error('CPU trainer stopped.')); for (const thread of this.threads)
        thread.port.terminate(); this.threads = []; this.engine.destroy(); }
}
//# sourceMappingURL=parallel-wasm.js.map