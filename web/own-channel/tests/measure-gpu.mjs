import {WebGPUTrainer} from '../dist/gpu.mjs';
import {Sampler, Grokking, ShortGrokking} from '../dist/model.mjs';
import {modelInfo} from '../dist/models.mjs';

export async function measureGPU(model, seed = modelInfo(model).seed) {
  const info = modelInfo(model), engine = await WebGPUTrainer.create(seed, {model}), sampler = new Sampler(seed), criterion = info.plateauEpochs ? new ShortGrokking() : new Grokking(), history = [], start = performance.now();
  globalThis.grokkingMeasurement = {model, history};
  try {
    for (;;) {
      const row = await engine.evaluate(); row.phase = criterion.observe(row); history.push(row);
      if (engine.step >= 100000 || row.heldoutAccuracy >= .99) break;
      for (let n = 0; n < info.reportEvery; n += 10) await engine.train(Array.from({length:10}, () => sampler.sample().slice()));
    }
    return {model, seed, reportEvery:info.reportEvery, backend:'webgpu', info:engine.info(), elapsedSeconds:(performance.now()-start)/1000, criterion, history};
  } finally { engine.destroy(); }
}
