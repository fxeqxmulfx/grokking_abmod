export type ModelId = 'dense-rotary' | 'dense-rotary-wide';
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
  destroy(): void;
}
export interface TrainingOptions {
  model: ModelId; seed: number; backend: Backend; maxEpochs: number;
  captureHz: number; stopAtTarget: boolean;
}
export interface Interval {start: number | null; end: number | null; epochs: number}
export interface GrokkingEvent {
  epoch: number; delayed: boolean; plateauEpochs: number;
  perfectTrainEpochs: number; riseEpochs: number | null;
}
export interface Criterion {
  plateau: Interval; perfect: Interval; event: GrokkingEvent | null;
}
export type WorkerRequest =
  | {type: 'initialize'; options: TrainingOptions}
  | {type: 'train' | 'step' | 'pause'}
  | {type: 'configure'; maxEpochs: number; stopAtTarget: boolean}
  | {type: 'capture'; hz: number};
export type WorkerResponse =
  | {type: 'ready'; info: EngineInfo}
  | {type: 'status'; status: string}
  | {type: 'weights'; step: number; weights: Weights; activeMs: number}
  | {type: 'metrics'; row: Metrics; criterion: Criterion}
  | {type: 'error'; message: string};
export interface Point {x: number; y: number}
export interface NetworkView extends Point {zoom: number}
export interface NetworkNode extends Point {channel: number; value?: number; label?: string}
export interface NetworkEdge {from: Point; to: Point; weight: number; label: string}
export const errorText = (error: unknown): string => error instanceof Error ? error.message : String(error);
export function send(worker: Worker | null, message: WorkerRequest): void {worker?.postMessage(message);}
export function backend(value: string): Backend {
  if (!BACKENDS.includes(value as Backend)) throw new Error('Unknown training engine.');
  return value as Backend;
}
