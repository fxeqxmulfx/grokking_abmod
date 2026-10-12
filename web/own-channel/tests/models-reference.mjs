import {P} from '../dist/model.mjs';
import {modelInfo, predictModel, inputMatrix, meshStride, readoutIndex} from '../dist/models.mjs';

// Scalar f64 backpropagation, with an explicit derivative for every mesh edge
// and each of the three uses of the tied embedding. No cached gain products.
export function modelLossAndGradient(weights, data, indices, id) {
  const model = modelInfo(id), W = model.width, gradient = new Float64Array(weights.length), projected = inputMatrix(weights, id);
  let loss = 0;
  for (const sample of indices) {
    const a = data[sample * 3], b = data[sample * 3 + 1], target = data[sample * 3 + 2];
    const prediction = predictModel(weights, a, b, id, projected), {hidden, logits} = prediction;
    const maximum = Math.max(...logits), denominator = logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0);
    loss += (Math.log(denominator) + maximum - logits[target]) / indices.length;
    const dl = logits.map((value, k) => (Math.exp(value - maximum) / denominator - Number(k === target)) / indices.length);
    let dh = new Float64Array(W);
    for (let c = 0; c < W; c++) for (let k = 0; k < P; k++) {
      const index = readoutIndex(id, c, k);
      gradient[index] += hidden[c] * dl[k]; dh[c] += weights[index] * dl[k];
    }
    if (id === 'concat') {
      const activations = [prediction.inputHidden];
      for (let layer = 0; layer < model.gains; layer++) activations.push(activations[layer].map((value, c) => Math.max(0, value * weights[model.gain + layer * W + c])));
      for (let layer = model.gains - 1; layer >= 0; layer--) for (let c = 0; c < W; c++) {
        const index = model.gain + layer * W + c;
        if (activations[layer][c] * weights[index] <= 0) dh[c] = 0;
        gradient[index] += dh[c] * activations[layer][c]; dh[c] *= weights[index];
      }
      for (let c = 0; c < W; c++) if (prediction.rawInput[c] <= 0) dh[c] = 0;
    } else {
      dh = normBackward(prediction.finalBefore, prediction.final, dh);
      for (let block = model.blocks - 1; block >= 0; block--) {
        const stage = prediction.stages[block]; let branch = dh.slice();
        if (model.expansion) {
          const at = model.gain + block * 2 * W * stage.expanded.length, expandedGradient = new Float64Array(stage.expanded.length), normalizedGradient = new Float64Array(W);
          for (let j = 0; j < stage.expanded.length; j++) for (let c = 0; c < W; c++) {
            const index = at + W * stage.expanded.length + j * W + c;
            gradient[index] += stage.activated[j] * dh[c]; expandedGradient[j] += weights[index] * dh[c];
          }
          for (let j = 0; j < expandedGradient.length; j++) if (stage.expanded[j] <= 0) expandedGradient[j] = 0;
          for (let c = 0; c < W; c++) for (let j = 0; j < expandedGradient.length; j++) {
            const index = at + c * expandedGradient.length + j;
            gradient[index] += stage.normalized.values[c] * expandedGradient[j]; normalizedGradient[c] += weights[index] * expandedGradient[j];
          }
          branch = normBackward(stage.before, stage.normalized, normalizedGradient);
          dh = dh.map((value, c) => value + branch[c]);
          continue;
        }
        for (let layer = model.meshDepth - 1; layer >= 0; layer--) {
          const flat = block * model.meshDepth + layer, stride = meshStride(model, flat), inputs = stage.values[layer], next = new Float64Array(W);
          for (let c = 0; c < W; c++) for (let parent = 0; parent < model.parents; parent++) {
            const source = (c + parent * stride) % W, index = model.gain + (flat * W + c) * model.parents + parent;
            const value = inputs[source], square = model.activation === 'square';
            gradient[index] += (layer ? square ? value * value : Math.max(0, value) : value) * branch[c];
            next[source] += weights[index] * branch[c] * (layer ? square ? 2 * value : Number(value > 0) : 1);
          }
          branch = next;
        }
        branch = normBackward(stage.before, stage.normalized, branch);
        dh = dh.map((value, c) => value + branch[c]);
      }
    }
    if (model.rotaryEmbedding) {
      for (let c = 0; c < W; c++) {
        gradient[a * W + c] += dh[c];
        gradient[b * W + c] += (c % 2 ? -1 : 1) * dh[c ^ 1];
      }
    } else if (model.tiedEmbedding) {
      for (const [operand, residue] of [[0, a], [1, b]]) for (let j = 0; j < W; j++) for (let c = 0; c < W; c++) {
        const index = model.projection + (operand * W + j) * W + c;
        gradient[residue * W + j] += weights[index] * dh[c];
        gradient[index] += weights[residue * W + j] * dh[c];
      }
    } else {
      for (let c = 0; c < W; c++) { gradient[a * W + c] += dh[c]; gradient[(P + b) * W + c] += dh[c]; }
    }
  }
  return {loss, gradient};
}

function normBackward(input, normalized, gradient) {
  const dot = input.reduce((sum, value, c) => sum + value * gradient[c], 0), r = normalized.inverse;
  return gradient.map((value, c) => r * value - input[c] * r ** 3 * dot / input.length);
}
