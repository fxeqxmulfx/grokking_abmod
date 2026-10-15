import type { Backend, ModelId, OptimizerSettings, Preset, TrainingEngine } from './types.ts';
export declare function createEngine(seed: number, backend?: Backend, model?: ModelId, preset?: Preset, custom?: OptimizerSettings): Promise<TrainingEngine>;
