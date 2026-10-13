import type { Backend, ModelId, TrainingEngine } from './types.ts';
export declare function createEngine(seed: number, backend?: Backend, model?: ModelId): Promise<TrainingEngine>;
