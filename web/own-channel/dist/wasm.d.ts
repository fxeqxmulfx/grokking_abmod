import type { Sample, Metrics, TrainerOptions, EngineInfo, EngineCheckpoint } from './types.ts';
export interface TrainerExports {
    memory: WebAssembly.Memory;
    weights: WebAssembly.Global;
    gradient: WebAssembly.Global;
    moment1: WebAssembly.Global;
    moment2: WebAssembly.Global;
    samples: WebAssembly.Global;
    metrics: WebAssembly.Global;
    logits: WebAssembly.Global;
    data: WebAssembly.Global;
    statistics: WebAssembly.Global;
    prepare(): void;
    gradient_batch(rows: number): void;
    gradient_batch_scaled(rows: number, scale: number): void;
    add_gradient(source: number): void;
    apply_update(rate: number, decay: number, correction1: number, correction2: number): void;
    step(rows: number, rate: number, decay: number, correction1: number, correction2: number): void;
    evaluate(): void;
    evaluate_range(first: number, end: number): void;
    forward(id: number): void;
}
import { modelInfo } from './models.ts';
export declare class WASMTrainer {
    static create(seed?: number, options?: TrainerOptions): Promise<WASMTrainer>;
    instance: WebAssembly.Instance;
    exports: TrainerExports;
    scalar: boolean;
    model: ReturnType<typeof modelInfo>;
    module: WebAssembly.Module;
    weights: Float32Array<ArrayBufferLike>;
    gradients: Float32Array<ArrayBufferLike>;
    samples: Uint32Array<ArrayBufferLike>;
    metrics: Float32Array<ArrayBufferLike>;
    logits: Float32Array<ArrayBufferLike>;
    data: Uint32Array<ArrayBuffer>;
    step: number;
    b1: number;
    b2: number;
    learningRate: number;
    decay: number;
    channel: MessageChannel;
    yieldTask: () => Promise<void>;
    constructor(instance: WebAssembly.Instance, seed: number, options: TrainerOptions & {
        scalar: boolean;
        model: NonNullable<TrainerOptions['model']>;
    });
    train(samples: Sample[], { gradientsOnly }?: {
        gradientsOnly?: boolean | undefined;
    }): Promise<void>;
    applyGradient(): void;
    evaluate(): Promise<Metrics>;
    readWeights(): Promise<Float32Array<ArrayBuffer>>;
    readGradients(): Promise<Float32Array<ArrayBuffer>>;
    readLogits(ids?: number[]): Promise<Float32Array<ArrayBuffer>[]>;
    info(): EngineInfo;
    checkpoint(): EngineCheckpoint;
    restore(checkpoint: EngineCheckpoint): void;
    destroy(): void;
}
export declare function validateSubmission(samples: Sample[], gradientsOnly?: boolean): void;
