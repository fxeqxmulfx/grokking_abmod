import {P, Random} from './model.ts';
import {OPTIMIZER_PROFILES} from './optimizer-profiles.ts';
import type {OptimizerProfile} from './optimizer-profiles.ts';
import {PRESETS} from './types.ts';
import type {BuiltinModelId, ModelId, OptimizerSettings, Preset, Vector, Weights} from './types.ts';

export interface ModelInfo extends OptimizerProfile {
  id: ModelId; title: string; subtitle: string; width: number; seed: number;
  reportEvery: number; parameters: number; embedding: number; gain: number;
}
function dense(id: ModelId, width: number): ModelInfo {
  return {id, title: 'Rotate90', subtitle: 'Dense + ReLU', width, seed: 11,
    ...(OPTIMIZER_PROFILES[width]?.fast ?? {learningRate: .01, weightDecay: 1, nativeSeeds: [], nativeTest99Epochs: [], nativeBudget: 10000}),
    reportEvery: 10, parameters: P * width + width * width, embedding: P * width, gain: P * width};
}
export const MODELS: Record<BuiltinModelId, ModelInfo> = {
  'dense-rotary-16': dense('dense-rotary-16', 16),
  'dense-rotary-32': dense('dense-rotary-32', 32),
  'dense-rotary-64': dense('dense-rotary-64', 64),
  'dense-rotary': dense('dense-rotary', 128),
  'dense-rotary-256': dense('dense-rotary-256', 256),
};
export function modelId(id: string): ModelId {
  if (Object.hasOwn(MODELS, id)) return id as BuiltinModelId;
  const match = /^dense-rotary-([1-9]\d*)$/.exec(id);
  if (!match) throw new Error('Unknown model.');
  return modelForWidth(Number(match[1]));
}
export function validateWidth(width: number): void {
  if (!Number.isInteger(width) || width < 2 || width > 2048 || width % 2 !== 0) throw new Error('Width must be an even integer between 2 and 2048.');
}
export function modelForWidth(width: number): ModelId {
  validateWidth(width);
  return width === 128 ? 'dense-rotary' : `dense-rotary-${width}`;
}
export function modelInfo(id: ModelId): ModelInfo {
  if (Object.hasOwn(MODELS, id)) return MODELS[id as BuiltinModelId];
  const canonical = modelId(id), width = Number(canonical.slice('dense-rotary-'.length));
  return canonical === 'dense-rotary' ? MODELS[canonical] : dense(canonical, width);
}
export function modelPresets(id: ModelId): Preset[] {return PRESETS.filter(preset => OPTIMIZER_PROFILES[modelInfo(id).width]?.[preset]);}
export function optimizerProfile(id: ModelId, preset: Preset = 'fast'): OptimizerProfile {
  const profile = OPTIMIZER_PROFILES[modelInfo(id).width]?.[preset];
  if (!profile) throw new Error('Unknown optimizer preset.');
  return profile;
}
export function trainingOptimizer(id: ModelId, preset: Preset = 'fast', custom?: OptimizerSettings): OptimizerSettings {
  const {learningRate, weightDecay} = custom ?? optimizerProfile(id, preset);
  if (!Number.isFinite(learningRate) || !Number.isFinite(Math.fround(learningRate)) || Math.fround(learningRate) <= 0) throw new Error('Learning rate must be a positive finite number.');
  if (!Number.isFinite(weightDecay) || !Number.isFinite(Math.fround(weightDecay)) || weightDecay < 0) throw new Error('Weight decay must be a nonnegative finite number.');
  return {learningRate, weightDecay};
}

// Box–Muller uses the native SplitMix64 stream and rounds each stored value to
// f32. E consumes the same number of draws as the old uniform initializer,
// leaving every Kaiming-normal A parameter unchanged.
export function initialModelWeights(seed = 11, id: ModelId = 'dense-rotary'): Weights {
  const {width, embedding, parameters} = modelInfo(id);
  const random = new Random(BigInt(seed) ^ 0x73696d706c650004n), weights = new Float32Array(parameters);
  for (const [start, end, variance] of [[0, embedding, 1 / width], [embedding, parameters, 2 / width]]) {
    const deviation = Math.sqrt(variance);
    for (let i = start; i < end; i += 2) {
      const radius = Math.sqrt(-2 * Math.log(1 - random.unit())), angle = 2 * Math.PI * random.unit();
      weights[i] = Math.fround(deviation * radius * Math.cos(angle));
      weights[i + 1] = Math.fround(deviation * radius * Math.sin(angle));
    }
  }
  return weights;
}
export function inputMatrix(weights: Vector, id: ModelId = 'dense-rotary'): Float64Array<ArrayBuffer> {
  const W = modelInfo(id).width, result = new Float64Array(2 * P * W);
  result.set(weights.subarray(0, P * W));
  for (let k = 0; k < P; k++) for (let c = 0; c < W; c++) result[(P + k) * W + c] = (c % 2 ? 1 : -1) * weights[k * W + (c ^ 1)];
  return result;
}
export function readoutIndex(id: ModelId, channel: number, answer: number): number {return answer * modelInfo(id).width + channel;}
export interface Normalized {inverse: number; values: Float64Array<ArrayBuffer>}
export function rms(values: Float64Array<ArrayBuffer>): Normalized {
  const inverse = 1 / Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length + 1e-5);
  return {inverse, values: values.map(value => value * inverse)};
}
export interface Stage {
  before: Float64Array<ArrayBuffer>; normalized: Normalized;
  values: [Float64Array<ArrayBuffer>, Float64Array<ArrayBuffer>]; after: Float64Array<ArrayBuffer>;
}
export interface Prediction {
  answer: number; hidden: Float64Array<ArrayBuffer>; logits: Float64Array<ArrayBuffer>;
  inputHidden: Float64Array<ArrayBuffer>; rawInput: Float64Array<ArrayBuffer>;
  stages: [Stage]; finalBefore: Float64Array<ArrayBuffer>; final: Normalized;
}
export function predictModel(weights: Vector, a: number, b: number, id: ModelId = 'dense-rotary', projected?: Float64Array<ArrayBuffer>): Prediction {
  const {width: W, gain} = modelInfo(id), input = projected ?? inputMatrix(weights, id);
  const rawInput = Float64Array.from({length: W}, (_, c) => input[a * W + c] + input[(P + b) * W + c]);
  const normalized = rms(rawInput), z = new Float64Array(W);
  for (let i = 0; i < W; i++) for (let c = 0; c < W; c++) z[c] += normalized.values[i] * weights[gain + i * W + c];
  const finalBefore = rawInput.map((value, c) => value + Math.max(0, z[c])), final = rms(finalBefore);
  const hidden = final.values, logits = new Float64Array(P);
  for (let c = 0; c < W; c++) for (let k = 0; k < P; k++) logits[k] += hidden[c] * weights[k * W + c];
  return {answer: logits.indexOf(Math.max(...logits)), hidden, logits, inputHidden: rawInput, rawInput,
    stages: [{before: rawInput, normalized, values: [normalized.values, z], after: finalBefore}], finalBefore, final};
}
