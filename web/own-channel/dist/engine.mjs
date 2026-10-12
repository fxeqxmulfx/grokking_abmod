import {WebGPUTrainer} from './gpu.mjs';
import {WASMTrainer} from './wasm.mjs';

export async function createEngine(seed, backend = 'auto', model = 'concat') {
  if (!['auto', 'wasm', 'gpu'].includes(backend)) throw new Error('Unknown training engine.');
  if (backend === 'wasm') return WASMTrainer.create(seed, {model});
  if (backend === 'gpu') return WebGPUTrainer.create(seed, {model});
  let note;
  try {
    const gpu = await WebGPUTrainer.create(seed, {model});
    if (!gpu.info().fallback) return gpu;
    gpu.destroy(); note = 'The GPU adapter is software-based; using WASM on the CPU.';
  } catch (error) { note = `WebGPU could not start; using WASM on the CPU. ${error.message}`; }
  const cpu = await WASMTrainer.create(seed, {model});
  const info = cpu.info(); cpu.info = () => ({...info, note});
  return cpu;
}
