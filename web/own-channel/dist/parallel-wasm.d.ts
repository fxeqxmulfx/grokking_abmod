import { cpuMemoryLayout } from './cpu-memory.ts';
import type { CPUThread } from './cpu-protocol.ts';
import type { TrainerOptions, Sample, Metrics, EngineInfo } from './types.ts';
import { WASMTrainer } from './wasm.ts';
export interface ParallelOptions extends TrainerOptions {
    threads: number;
    workerFactory?: () => CPUThread;
}
export declare class ParallelWASMTrainer {
    readonly engine: WASMTrainer;
    readonly layout: ReturnType<typeof cpuMemoryLayout>;
    readonly memory: WebAssembly.Memory;
    readonly count: number;
    static create(seed: number | undefined, options: ParallelOptions): Promise<ParallelWASMTrainer>;
    private threads;
    private control;
    private job;
    private busy;
    private failure;
    constructor(engine: WASMTrainer, layout: ReturnType<typeof cpuMemoryLayout>, memory: WebAssembly.Memory, count: number);
    get step(): number;
    private fail;
    private send;
    private begin;
    private phase;
    private acquire;
    train(samples: Sample[], { gradientsOnly }?: {
        gradientsOnly?: boolean | undefined;
    }): Promise<void>;
    evaluate(): Promise<Metrics>;
    readWeights(): Promise<Float32Array<ArrayBuffer>>;
    readGradients(): Promise<Float32Array<ArrayBuffer>>;
    readLogits(ids?: number[]): Promise<Float32Array<ArrayBuffer>[]>;
    info(): EngineInfo;
    destroy(): void;
}
