import type { Backend, ModelId, Preset, TrainingEngine } from './types.ts';
export declare function createEngine(seed: number, backend?: Backend, model?: ModelId, preset?: Preset): Promise<TrainingEngine>;
