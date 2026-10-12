import {P, W, INPUT, GAIN, PARAMETERS, predict} from '../dist/model.mjs';

// Independent scalar f64 reference for the handwritten WGSL and WAT trainers.
export function lossAndGradient(weights, data, indices) {
  const gradient = new Float64Array(PARAMETERS);
  let loss = 0;
  for (const id of indices) {
    const a = data[id * 3], b = data[id * 3 + 1], target = data[id * 3 + 2];
    const {hidden, logits} = predict(weights, a, b);
    const maximum = Math.max(...logits), denominator = logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0);
    loss += (Math.log(denominator) + maximum - logits[target]) / indices.length;
    const dl = logits.map((value, k) => (Math.exp(value - maximum) / denominator - Number(k === target)) / indices.length);
    for (let c = 0; c < W; c++) {
      let dh = 0;
      for (let k = 0; k < P; k++) { gradient[INPUT + c * P + k] += hidden[c] * dl[k]; dh += weights[INPUT + c * P + k] * dl[k]; }
      // Reverse the explicit sequence of four ReLU layers, rather than using
      // the product and excluded products cached by the GPU implementation.
      const activations = [Math.max(0, weights[a * W + c] + weights[(P + b) * W + c])];
      for (let layer = 0; layer < 4; layer++) activations.push(Math.max(0, activations[layer] * weights[GAIN + layer * W + c]));
      for (let layer = 3; layer >= 0; layer--) {
        const index = GAIN + layer * W + c;
        if (activations[layer] * weights[index] <= 0) dh = 0;
        gradient[index] += dh * activations[layer]; dh *= weights[index];
      }
      if (weights[a * W + c] + weights[(P + b) * W + c] > 0) { gradient[a * W + c] += dh; gradient[(P + b) * W + c] += dh; }
    }
  }
  return {loss, gradient};
}

export function adamStep(weights, gradient, moments, step) {
  const rate = .001 * Math.min(step / 10, 1);
  for (let i = 0; i < weights.length; i++) {
    moments[i * 2] = .9 * moments[i * 2] + .1 * gradient[i];
    moments[i * 2 + 1] = .98 * moments[i * 2 + 1] + .02 * gradient[i] ** 2;
    weights[i] -= rate * (moments[i * 2] / (1 - .9 ** step) / (Math.sqrt(moments[i * 2 + 1] / (1 - .98 ** step)) + 1e-8) + .1 * weights[i]);
  }
}
