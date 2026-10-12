import {WebGPUTrainer} from '../dist/gpu.mjs';
import {initialModelWeights, modelInfo, inputMatrix, predictModel} from '../dist/models.mjs';
import {dataset, Sampler, P} from '../dist/model.mjs';
import {modelLossAndGradient} from './models-reference.mjs';
import {adamStep} from './reference.mjs';

function close(actual, expected, tolerance, label) {
  if (actual.length !== expected.length || !actual.every(Number.isFinite) || !expected.every(Number.isFinite)) throw new Error(`${label}: invalid numerical arrays`);
  let error = 0, index = 0;
  for (let i = 0; i < actual.length; i++) if (Math.abs(actual[i] - expected[i]) > error) { error = Math.abs(actual[i] - expected[i]); index = i; }
  if (!Number.isFinite(error) || error > tolerance) throw new Error(`${label}: error ${error} at ${index}, tolerance ${tolerance}`);
  return error;
}

export async function verifyModelGPU(model = 'concat') {
  const weights = initialModelWeights(7, model), data = dataset(), info = modelInfo(model), results = [];
  const engine = await WebGPUTrainer.create(7, {model});
  try {
    const logits = await engine.readLogits(), projected = inputMatrix(weights, model);
    for (const id of [0, 19, 4655, 4656, 9311]) results.push({check: `logits ${id}`, error: close(logits.slice(id * P, (id + 1) * P), predictModel(weights, data[id * 3], data[id * 3 + 1], model, projected).logits, 8e-6, 'logits')});
    for (const n of [1, 37, 48, 512]) {
      const sample = new Sampler().sample().slice(0, n);
      await engine.train([sample], {gradientsOnly: true});
      const actual = await engine.readGradients(), expected = modelLossAndGradient(weights, data, sample, model).gradient;
      const boundary = info.tiedEmbedding ? info.embedding : info.readout;
      if (info.meshDepth) for (const [start, end] of [[0, boundary], [boundary, info.gain], [info.gain, info.parameters]].filter(([start, end]) => end > start)) {
        const reference = expected.subarray(start, end), scale = reference.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 1e-4);
        const error = close(actual.subarray(start, end), reference, 8e-6 * scale, 'gradients');
        results.push({check: `parameters ${start}..${end}, batch ${n}`, error, relativeError:error / scale});
      } else results.push({check: `all ${info.parameters} gradients, batch ${n}`, error:close(actual, expected, 1e-5, 'gradients')});
    }
    const expected = Float64Array.from(weights), moments = new Float64Array(info.parameters * 2), sampler = new Sampler();
    for (let step = 1; step <= 3; step++) {
      const sample = sampler.sample().slice();
      adamStep(expected, modelLossAndGradient(expected, data, sample, model).gradient, moments, step);
      await engine.train([sample]);
      results.push({check: `AdamW ${step}`, error: close(await engine.readWeights(), expected, 3e-5, 'AdamW')});
    }
    const row = await engine.evaluate();
    if (!Number.isFinite(row.trainLoss) || !Number.isFinite(row.heldoutLoss)) throw new Error('Nonfinite metrics');
    results.push({check: 'complete-split metrics', row});
  } finally { engine.destroy(); }
  const variant = weights.slice();
  for (let i = info.gain; i < variant.length; i++) variant[i] = .4 + .05 * (i % 11);
  variant[info.gain + 9] = -.5; variant[info.gain + 10] = 0;
  const modified = await WebGPUTrainer.create(7, {model, weights: variant});
  try {
    const sample = new Sampler().sample().slice(0, 37);
    await modified.train([sample], {gradientsOnly: true});
    results.push({check: 'nontrivial independent gains/edges', error: close(await modified.readGradients(), modelLossAndGradient(variant, data, sample, model).gradient, 2e-5, 'variant')});
  } finally { modified.destroy(); }
  return results;
}
