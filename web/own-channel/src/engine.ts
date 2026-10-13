import {BACKENDS, errorText} from './types.ts';
import type {Backend, ModelId, TrainingEngine} from './types.ts';
import {WASMTrainer} from './wasm.ts';
import {ParallelWASMTrainer} from './parallel-wasm.ts';

export async function createEngine(seed: number, backend: Backend = 'auto', model: ModelId = 'dense-rotary'): Promise<TrainingEngine> {
  if (!BACKENDS.includes(backend)) throw new Error('Unknown training engine.');
  const scalar = backend.startsWith('scalar'), requireSIMD = backend.startsWith('simd');
  const available = globalThis.crossOriginIsolated && typeof SharedArrayBuffer === 'function' && typeof Worker === 'function';
  const cores = typeof navigator === 'object' ? navigator.hardwareConcurrency || 1 : 1;
  // Four training threads win on the measured six-core CPU. Reserve capacity
  // for the UI/render worker; smaller devices use two or one training thread.
  const requested = backend === 'auto' ? (cores >= 6 ? 4 : cores >= 4 ? 2 : 1) : Number(backend.split('-')[1] ?? 1);
  let note: string | undefined;
  if (requested > 1 && available) {
    try {return await ParallelWASMTrainer.create(seed, {model, scalar, requireSIMD, threads: requested});}
    catch (error) {note = `Parallel WASM could not start; using one CPU thread. ${errorText(error)}`;}
  } else if (requested > 1) note = 'Shared memory is unavailable; using one CPU thread.';
  const engine = await WASMTrainer.create(seed, {model, scalar, requireSIMD});
  if (note) {const info = engine.info(); engine.info = () => ({...info, note});}
  return engine;
}
