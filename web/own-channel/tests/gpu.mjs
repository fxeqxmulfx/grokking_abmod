import {WebGPUTrainer} from '../dist/gpu.mjs';
import {initialWeights, dataset, Sampler, GAIN, PARAMETERS, P, W, predict} from '../dist/model.mjs';
import {lossAndGradient, adamStep} from './reference.mjs';

function close(name, actual, expected, tolerance) {
  const error = Math.max(...actual.map((value, i) => Math.abs(value - expected[i])));
  if (!Number.isFinite(error) || error > tolerance) throw new Error(`${name}: max error ${error}, tolerance ${tolerance}`);
  return error;
}

export async function verifyGPU() {
  const data = dataset(), weights = initialWeights(), results = [];
  let engine = await WebGPUTrainer.create();
  try {
    const logits = await engine.readLogits();
    for (const id of [0, 19, 4655, 4656, 9311]) results.push({check: `logits row ${id}`, error: close('logits', logits.slice(id * P, (id + 1) * P), predict(weights, data[id * 3], data[id * 3 + 1]).logits, 3e-6)});
    for (const n of [1, 37, 48, 512]) {
      const sample = new Sampler().sample().slice(0, n), reference = lossAndGradient(weights, data, sample);
      await engine.train([sample], {gradientsOnly: true}); const gradient = await engine.readGradients();
      results.push({check: `all ${PARAMETERS} gradients, batch ${n}`, error: close('gradients', gradient, reference.gradient, 3e-6)});
    }
    const expected = Float64Array.from(weights), moments = new Float64Array(PARAMETERS * 2), sampler = new Sampler();
    for (let step = 1; step <= 3; step++) {
      const sample = sampler.sample().slice(); const {gradient} = lossAndGradient(expected, data, sample);
      adamStep(expected, gradient, moments, step); await engine.train([sample]);
      results.push({check: `AdamW step ${step}`, error: close('AdamW weights', await engine.readWeights(), expected, 1e-5)});
    }
    const metrics = await engine.evaluate();
    const snapshot = await engine.readWeights();
    const training = lossAndGradient(snapshot, data, Uint32Array.from({length: 4656}, (_, i) => i));
    if (Math.abs(training.loss - metrics.trainLoss) > 1e-5) throw new Error('Complete-split training loss disagrees with reference');
    results.push({check: 'complete-split metrics', loss: metrics.trainLoss});
  } finally { engine.destroy(); }
  const variant = weights.slice();
  for (let c = 0; c < W; c++) for (let layer = 0; layer < 4; layer++) variant[GAIN + layer * W + c] = .7 + layer * .3;
  variant[GAIN + 9] = -.5; variant[GAIN + W + 10] = 0;
  engine = await WebGPUTrainer.create(7, {weights: variant});
  try {
    const sample = new Sampler().sample().slice(0, 37), reference = lossAndGradient(variant, data, sample);
    await engine.train([sample], {gradientsOnly: true});
    results.push({check: 'untied unequal, zero and negative gains', error: close('variant gradient', await engine.readGradients(), reference.gradient, 3e-6)});
  } finally { engine.destroy(); }
  return results;
}
