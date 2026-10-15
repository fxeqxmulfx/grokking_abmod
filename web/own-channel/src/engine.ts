import {BACKENDS, errorText} from './types.ts';
import type {Backend, ModelId, OptimizerSettings, Preset, TrainingEngine} from './types.ts';
import {modelInfo, trainingOptimizer} from './models.ts';
import {WASMTrainer} from './wasm.ts';
import {ParallelWASMTrainer} from './parallel-wasm.ts';

export async function createEngine(seed: number, backend: Backend = 'auto', model: ModelId = 'dense-rotary', preset: Preset = 'fast', custom?: OptimizerSettings): Promise<TrainingEngine> {
  if (!BACKENDS.includes(backend)) throw new Error('Unknown training engine.');
  const {learningRate, weightDecay: decay} = trainingOptimizer(model, preset, custom);
  const scalarWidth = modelInfo(model).width % 4 !== 0;
  const scalar = backend.startsWith('scalar') || scalarWidth, requireSIMD = backend.startsWith('simd') && !scalarWidth;
  const available = globalThis.crossOriginIsolated && typeof SharedArrayBuffer === 'function' && typeof Worker === 'function';
  const cores = typeof navigator === 'object' ? navigator.hardwareConcurrency || 1 : 1;
  // Four training threads win on the measured six-core CPU. Reserve capacity
  // for the UI/render worker; smaller devices use two or one training thread.
  const requested = backend === 'auto' ? (cores >= 6 ? 4 : cores >= 4 ? 2 : 1) : Number(backend.split('-')[1] ?? 1);
  let note: string | undefined = scalarWidth && !backend.startsWith('scalar') ? 'This width uses scalar WASM; SIMD requires a multiple of four channels.' : undefined;
  const describe = (engine: TrainingEngine) => {
    if (note) {const info = engine.info(); engine.info = () => ({...info, note});}
    return engine;
  };
  if (requested > 1 && available) {
    try {return describe(await ParallelWASMTrainer.create(seed, {model, scalar, requireSIMD, threads: requested, learningRate, decay}));}
    catch (error) {note = [note, `Parallel WASM could not start; using one CPU thread. ${errorText(error)}`].filter(Boolean).join(' ');}
  } else if (requested > 1) note = [note, 'Shared memory is unavailable; using one CPU thread.'].filter(Boolean).join(' ');
  const engine = await WASMTrainer.create(seed, {model, scalar, requireSIMD, learningRate, decay});
  return describe(engine);
}
