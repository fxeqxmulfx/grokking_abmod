import type { ModelId, Vector, Weights } from './types.ts';
export interface ModelInfo {
    id: ModelId;
    title: string;
    subtitle: string;
    width: number;
    seed: number;
    reportEvery: number;
    parameters: number;
    embedding: number;
    gain: number;
}
export declare const MODELS: Record<ModelId, ModelInfo>;
export declare function modelId(id: string): ModelId;
export declare function modelInfo(id: ModelId): ModelInfo;
export declare function initialModelWeights(seed?: number, id?: ModelId): Weights;
export declare function inputMatrix(weights: Vector, id?: ModelId): Float64Array<ArrayBuffer>;
export declare function readoutIndex(id: ModelId, channel: number, answer: number): number;
export interface Normalized {
    inverse: number;
    values: Float64Array<ArrayBuffer>;
}
export declare function rms(values: Float64Array<ArrayBuffer>): Normalized;
export interface Stage {
    before: Float64Array<ArrayBuffer>;
    normalized: Normalized;
    values: [Float64Array<ArrayBuffer>, Float64Array<ArrayBuffer>];
    after: Float64Array<ArrayBuffer>;
}
export interface Prediction {
    answer: number;
    hidden: Float64Array<ArrayBuffer>;
    logits: Float64Array<ArrayBuffer>;
    inputHidden: Float64Array<ArrayBuffer>;
    rawInput: Float64Array<ArrayBuffer>;
    stages: [Stage];
    finalBefore: Float64Array<ArrayBuffer>;
    final: Normalized;
}
export declare function predictModel(weights: Vector, a: number, b: number, id?: ModelId, projected?: Float64Array<ArrayBuffer>): Prediction;
