import {P} from '../dist/model.js';
import {modelInfo, predictModel, inputMatrix} from '../dist/models.js';
import type {Normalized} from '../dist/models.js';
import type {ModelId, Vector, Sample} from '../dist/types.js';

// Independent scalar f64 backpropagation. Explicitly accumulate the readout,
// a-embedding and rotated b-embedding derivatives into one shared E.
export function modelLossAndGradient(weights: Vector, data: Uint32Array, indices: Sample, id: ModelId) {
  const {width: W, gain} = modelInfo(id), gradient = new Float64Array(weights.length), projected = inputMatrix(weights, id);
  let loss = 0;
  for (const sample of indices) {
    const a = data[sample * 3], b = data[sample * 3 + 1], target = data[sample * 3 + 2];
    const prediction = predictModel(weights, a, b, id, projected), {hidden, logits} = prediction;
    const maximum = Math.max(...logits), denominator = logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0);
    loss += (Math.log(denominator) + maximum - logits[target]) / indices.length;
    const dl = logits.map((value, k) => (Math.exp(value - maximum) / denominator - Number(k === target)) / indices.length);
    let dh = new Float64Array(W);
    for (let c = 0; c < W; c++) for (let k = 0; k < P; k++) {
      const index = k * W + c;
      gradient[index] += hidden[c] * dl[k]; dh[c] += weights[index] * dl[k];
    }
    dh = normBackward(prediction.finalBefore, prediction.final, dh);
    const stage = prediction.stages[0], dz = dh.map((value, c) => value * Number(stage.values[1][c] > 0)), dr = new Float64Array(W);
    for (let i = 0; i < W; i++) for (let c = 0; c < W; c++) {
      const index = gain + i * W + c;
      gradient[index] += stage.normalized.values[i] * dz[c]; dr[i] += weights[index] * dz[c];
    }
    const branch = normBackward(stage.before, stage.normalized, dr);
    dh = dh.map((value, c) => value + branch[c]);
    for (let c = 0; c < W; c++) {
      gradient[a * W + c] += dh[c];
      gradient[b * W + c] += (c % 2 ? -1 : 1) * dh[c ^ 1];
    }
  }
  return {loss, gradient};
}
function normBackward(input: Float64Array, normalized: Normalized, gradient: Float64Array<ArrayBuffer>) {
  const dot = input.reduce((sum, value, c) => sum + value * gradient[c], 0), r = normalized.inverse;
  return gradient.map((value, c) => r * value - input[c] * r ** 3 * dot / input.length);
}
