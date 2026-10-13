// Independent f64 AdamW oracle with one moment pair per unique parameter.
export function adamStep(weights: Float64Array, gradient: Float64Array, moments: Float64Array, step: number) {
  const rate = .001 * Math.min(step / 10, 1);
  for (let i = 0; i < weights.length; i++) {
    moments[i * 2] = .9 * moments[i * 2] + .1 * gradient[i];
    moments[i * 2 + 1] = .98 * moments[i * 2 + 1] + .02 * gradient[i] ** 2;
    weights[i] -= rate * (moments[i * 2] / (1 - .9 ** step) / (Math.sqrt(moments[i * 2 + 1] / (1 - .98 ** step)) + 1e-8) + .1 * weights[i]);
  }
}
