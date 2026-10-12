import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {WASMTrainer} from '../dist/wasm.mjs';
import {P, W, GAIN, PARAMETERS, Sampler, dataset, initialWeights, predict} from '../dist/model.mjs';
import {lossAndGradient, adamStep} from './reference.mjs';

function close(actual, expected, tolerance, name) {
  let error = 0;
  for (let i = 0; i < actual.length; i++) error = Math.max(error, Math.abs(actual[i] - expected[i]));
  assert.ok(Number.isFinite(error) && error <= tolerance, `${name}: max error ${error}, tolerance ${tolerance}`);
}
for (const variant of ['simd', 'scalar']) {
  const bytes = await readFile(new URL(`../dist/train-${variant}.wasm`, import.meta.url));
  const create = weights => WASMTrainer.create(7, {bytes, scalar: variant === 'scalar', weights});
  test(`handwritten ${variant} WASM has no imports and uses exactly 512 KiB`, async () => {
    const module = await WebAssembly.compile(bytes);
    assert.deepEqual(WebAssembly.Module.imports(module), []);
    const engine = await create();
    try { assert.equal(engine.info().memoryBytes, 524288); assert.deepEqual(await engine.readWeights(), initialWeights()); }
    finally { engine.destroy(); }
  });
  test(`${variant} exponential and logarithm approximations stay within f32 error`, async () => {
    const engine = await create();
    try {
      for (let i = 0; i <= 10000; i++) {
        const x = Math.fround(-87 * i / 10000), expected = Math.exp(x);
        assert.ok(Math.abs(engine.exports.exp_negative(x) - expected) / expected < 5e-7, `exp(${x})`);
        const y = Math.fround(1 + 96 * i / 10000);
        assert.ok(Math.abs(engine.exports.log_positive(y) - Math.log(y)) < 7e-7, `log(${y})`);
      }
    } finally { engine.destroy(); }
  });
  test(`${variant} logits and every gradient agree with independent f64 calculations`, async () => {
    const engine = await create(), weights = initialWeights(), data = dataset(), ids = [0, 19, 4655, 4656, 9311];
    try {
      const logits = await engine.readLogits(ids);
      ids.forEach((id, i) => close(logits[i], predict(weights, data[id * 3], data[id * 3 + 1]).logits, 3e-6, 'logits'));
      for (const n of [1, 37, 48, 512]) {
        const sample = new Sampler().sample().slice(0, n), reference = lossAndGradient(weights, data, sample);
        await engine.train([sample], {gradientsOnly: true});
        close(await engine.readGradients(), reference.gradient, 3e-6, `all ${PARAMETERS} gradients, batch ${n}`);
      }
    } finally { engine.destroy(); }
  });
  test(`${variant} keeps unequal/nonpositive gains independent`, async () => {
    const weights = initialWeights();
    for (let c = 0; c < W; c++) for (let l = 0; l < 4; l++) weights[GAIN + l * W + c] = .7 + l * .3;
    weights[GAIN + 9] = -.5; weights[GAIN + W + 10] = 0;
    const engine = await create(weights);
    try {
      const sample = new Sampler().sample().slice(0, 37);
      await engine.train([sample], {gradientsOnly: true});
      close(await engine.readGradients(), lossAndGradient(weights, dataset(), sample).gradient, 3e-6, 'untied gains');
    } finally { engine.destroy(); }
  });
  test(`${variant} AdamW updates and full-split metrics agree with f64 reference`, async () => {
    const engine = await create(), expected = Float64Array.from(initialWeights()), moments = new Float64Array(PARAMETERS * 2), sampler = new Sampler(), data = dataset();
    try {
      for (let step = 1; step <= 3; step++) {
        const sample = sampler.sample().slice();
        adamStep(expected, lossAndGradient(expected, data, sample).gradient, moments, step);
        await engine.train([sample]);
        close(await engine.readWeights(), expected, 1e-5, `AdamW step ${step}`);
      }
      const metrics = await engine.evaluate(), weights = await engine.readWeights();
      for (const [start, loss, accuracy] of [[0, metrics.trainLoss, metrics.trainAccuracy], [4656, metrics.heldoutLoss, metrics.heldoutAccuracy]]) {
        let expectedLoss = 0, correct = 0;
        for (let id = start; id < start + 4656; id++) {
          const {answer, logits} = predict(weights, data[id * 3], data[id * 3 + 1]);
          const target = data[id * 3 + 2], maximum = Math.max(...logits);
          expectedLoss += Math.log(logits.reduce((sum, x) => sum + Math.exp(x - maximum), 0)) + maximum - logits[target];
          correct += Number(answer === target);
        }
        assert.ok(Math.abs(loss - expectedLoss / 4656) < 1e-5);
        assert.ok(Math.abs(accuracy - correct / 4656) < 1e-7);
      }
    } finally { engine.destroy(); }
  });
}
