export const MODEL_IDS = ['dense-rotary-16', 'dense-rotary-32', 'dense-rotary-64', 'dense-rotary', 'dense-rotary-256'] as const;
export type ModelId = typeof MODEL_IDS[number];
// Optimizer presets: fast generalizes earliest on every native seed, plateau
// memorizes longest before generalizing, slow generalizes latest.
export const PRESETS = ['fast', 'plateau', 'slow'] as const;
export type Preset = typeof PRESETS[number];
export const BACKENDS = ['auto', 'simd', 'scalar', 'simd-2', 'simd-4', 'scalar-2', 'scalar-4'] as const;
export type Backend = typeof BACKENDS[number];
export type Weights = Float32Array<ArrayBuffer>;
export type Vector = Weights | Float64Array<ArrayBuffer>;
export type Sample = Uint32Array<ArrayBufferLike> | number[];
export type WeightMode = 'weight' | 'delta';
export interface Metrics {
  step: number; epoch: number; trainLoss: number; trainAccuracy: number;
  heldoutLoss: number; heldoutAccuracy: number; phase?: string;
}
export interface EngineInfo {
  description: string; backend: 'wasm'; threads: number;
  memoryBytes?: number; note?: string;
}
export interface TrainerOptions {
  model?: ModelId; scalar?: boolean; bytes?: Uint8Array<ArrayBuffer>;
  weights?: Weights; learningRate?: number; decay?: number;
  requireSIMD?: boolean; shared?: import('./cpu-memory.ts').SharedCPU;
}
export interface TrainingEngine {
  step: number;
  train(samples: Sample[], options?: {gradientsOnly?: boolean}): Promise<void>;
  evaluate(): Promise<Metrics>;
  readWeights(): Promise<Weights>;
  info(): EngineInfo;
  checkpoint(): EngineCheckpoint;
  restore(checkpoint: EngineCheckpoint): void;
  destroy(): void;
}
export interface EngineCheckpoint {
  model: ModelId; step: number; b1: number; b2: number;
  weights: Weights; moment1: Weights; moment2: Weights;
}
export interface TrainingCheckpoint {
  engine: EngineCheckpoint;
  sampler: {order: Uint32Array<ArrayBuffer>; position: number; low: number; high: number};
  criterion: Omit<import('./model.ts').ShortGrokking, 'observe'>;
  targetReached: boolean; activeMs: number; status: string;
}
export interface TrainingOptions {
  model: ModelId; preset: Preset; seed: number; backend: Backend; maxEpochs: number;
  captureHz: number; optimizer?: OptimizerSettings;
}
export interface OptimizerSettings {learningRate: number; weightDecay: number}
export interface Interval {start: number | null; end: number | null; epochs: number}
export interface GrokkingEvent {
  epoch: number; delayed: boolean; plateauEpochs: number;
  perfectTrainEpochs: number; riseEpochs: number | null;
}
export interface Criterion {
  plateau: Interval; perfect: Interval; event: GrokkingEvent | null;
}
export type WorkerRequest =
  | {type: 'initialize'; options: TrainingOptions; checkpoint?: TrainingCheckpoint}
  | {type: 'train' | 'step' | 'pause' | 'suspend'}
  | {type: 'configure'; maxEpochs: number}
  | {type: 'capture'; hz: number};
export type WorkerResponse =
  | {type: 'ready'; info: EngineInfo}
  | {type: 'status'; status: string}
  | {type: 'weights'; step: number; weights: Weights; activeMs: number}
  | {type: 'metrics'; row: Metrics; criterion: Criterion}
  | {type: 'suspended'; checkpoint?: TrainingCheckpoint}
  | {type: 'error'; message: string};
export interface Point {x: number; y: number}
export interface NetworkView extends Point {zoom: number}
export interface NetworkNode extends Point {channel: number; value?: number; label?: string}
export interface NetworkEdge {from: Point; to: Point; weight: number; label: string}
export const errorText = (error: unknown): string => error instanceof Error ? error.message : String(error);
export function send(worker: Worker | null, message: WorkerRequest): void {worker?.postMessage(message);}
export function preset(value: string): Preset {
  if (!PRESETS.includes(value as Preset)) throw new Error('Unknown optimizer preset.');
  return value as Preset;
}
export function backend(value: string): Backend {
  if (!BACKENDS.includes(value as Backend)) throw new Error('Unknown training engine.');
  return value as Backend;
}
