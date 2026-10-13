import {root} from '../project.ts';
import type {Vector, Weights, ModelId} from '../dist/types.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {WASMTrainer} from '../dist/wasm.js';
import {Sampler, dataset} from '../dist/model.js';
import {modelInfo, initialModelWeights, predictModel, inputMatrix} from '../dist/models.js';
import {modelLossAndGradient} from './models-reference.ts';
import {adamStep} from './reference.ts';

function close(actual: Vector, expected: Vector, tolerance: number, label: string) {
  let error = 0, index = 0;
  assert.equal(actual.length, expected.length);
  assert.ok(actual.every(Number.isFinite) && expected.every(Number.isFinite), `${label}: nonfinite values`);
  for (let i = 0; i < actual.length; i++) if (Math.abs(actual[i] - expected[i]) > error) { error = Math.abs(actual[i] - expected[i]); index = i; }
  assert.ok(Number.isFinite(error) && error <= tolerance, `${label}: error ${error} at ${index}, tolerance ${tolerance}`);
}

for (const model of ['dense-rotary', 'dense-rotary-wide'] as const) for (const variant of ['simd', 'scalar'] as const) {
  const bytes = await readFile(new URL(`dist/${model}-${variant}.wasm`, root)), info = modelInfo(model);
  const create = (weights?: Weights) => WASMTrainer.create(7, {model, bytes: new Uint8Array(bytes), scalar: variant === 'scalar', weights});
  test(`${model}/${variant} has no imports and matches the seeded initialization`, async () => {
    assert.deepEqual(WebAssembly.Module.imports(await WebAssembly.compile(bytes)), []);
    const engine = await create();
    try { assert.deepEqual(await engine.readWeights(), initialModelWeights(7, model)); assert.ok(engine.info().memoryBytes! <= (info.width === 1024 ? 20971520 : 1572864)); }
    finally { engine.destroy(); }
  });
  test(`${model}/${variant} logits and all gradients match f64 backpropagation`, async () => {
    const weights = initialModelWeights(7, model), engine = await create(weights), data = dataset(), projected = inputMatrix(weights, model);
    try {
      const ids = [0, 19, 4655, 4656, 9311], logits = await engine.readLogits(ids);
      ids.forEach((id, i) => close(logits[i], predictModel(weights, data[id * 3], data[id * 3 + 1], model, projected).logits, info.width === 1024 ? 1e-5 : 5e-6, 'logits'));
      for (const n of info.width === 1024 ? [1, 7, 48] : [1, 37, 48, 512]) {
        const sample = new Sampler().sample().slice(0, n);
        await engine.train([sample], {gradientsOnly: true});
        const actual = await engine.readGradients(), expected = modelLossAndGradient(weights, data, sample, model).gradient;
        close(actual, expected, 8e-6, `${info.parameters} gradients, batch ${n}`);
      }
    } finally { engine.destroy(); }
  });
  test(`${model}/${variant} signed dense connections and shared parameter accumulation`, async () => {
    const weights = initialModelWeights(7, model);
    for (let i = info.gain; i < weights.length; i++) weights[i] = .4 + .05 * (i % 11);
    weights[info.gain + 9] = -.5; weights[info.gain + 10] = 0;
    const engine = await create(weights), sample = new Sampler().sample().slice(0, 37);
    try {
      await engine.train([sample], {gradientsOnly: true});
      const actual = await engine.readGradients(), expected = modelLossAndGradient(weights, dataset(), sample, model).gradient;
      if (info.width === 1024) for (const [start, end] of [[0, info.embedding], [info.gain, info.parameters]]) {
        const reference = expected.subarray(start, end), scale = reference.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 1e-4);
        // Dense ReLU uses length-1024 FP32 dot products in both directions.
        // This coherent, non-Kaiming fixture amplifies their rounding in E.
        close(actual.subarray(start, end), reference, 8e-6 * Math.max(1, scale), 'nontrivial wide weights');
      } else close(actual, expected, 8e-6, 'nontrivial weights');
    } finally { engine.destroy(); }
  });
  test(`${model}/${variant} AdamW matches a single state per unique parameter`, async () => {
    const engine = await create(), expected = Float64Array.from(initialModelWeights(7, model)), moments = new Float64Array(info.parameters * 2), sampler = new Sampler(), data = dataset();
    try {
      for (let step = 1; step <= 3; step++) {
        const sample = sampler.sample().slice(0, info.width === 1024 ? 48 : 512);
        adamStep(expected, modelLossAndGradient(expected, data, sample, model).gradient, moments, step);
        await engine.train([sample]); close(await engine.readWeights(), expected, 2e-5, `AdamW step ${step}`);
      }
      const row = await engine.evaluate();
      assert.ok(Number.isFinite(row.trainLoss) && Number.isFinite(row.heldoutLoss));
    } finally { engine.destroy(); }
  });
}

test('Rotary roles preserve embedding norms and distinguish swapped nonidentical operands', () => {
  const weights = initialModelWeights(11, 'dense-rotary'), info = modelInfo('dense-rotary'), input = inputMatrix(weights, info.id);
  for (let k = 0; k < 97; k++) {
    let left = 0, right = 0;
    for (let c = 0; c < 128; c++) { left += input[k * 128 + c] ** 2; right += input[(97 + k) * 128 + c] ** 2; }
    assert.ok(Math.abs(left - right) < 1e-12);
  }
  assert.notDeepEqual(predictModel(weights, 19, 27, info.id, input).rawInput, predictModel(weights, 27, 19, info.id, input).rawInput);
  assert.equal(info.parameters, 28800);
});
test('Normal-E and Kaiming-A checkpoints match Rust byte-for-byte at both displayed widths', () => {
  for (const [model, hash] of [
    ['dense-rotary', 'b90448461caff3b6e04bd9ead0a3f7e7f055e8c1cd71ce19f1069186c74ae04b'],
    ['dense-rotary-wide', 'ea659d5b1596d3a019c3d2af55cb4c030244481b73011b0def3572ddc04d1e91'],
  ] as const) assert.equal(createHash('sha256').update(new Uint8Array(initialModelWeights(11, model).buffer)).digest('hex'), hash);
});

test('Dense derivatives include the pre-norm skip, ReLU and every shared E role', () => {
  const model = 'dense-rotary', info = modelInfo(model), weights = Float64Array.from(initialModelWeights(11, model)), data = dataset(11);
  const same = Array.from({length: 4656}, (_, i) => i).find(i => data[i * 3] === data[i * 3 + 1])!;
  const residue = data[same * 3], sample = [same, 0, 1], {gradient} = modelLossAndGradient(weights, data, sample, model);
  for (const index of [0, residue * 128, residue * 128 + 1, residue * 128 + 127, info.gain, info.gain + 127, info.gain + 127 * 128, info.parameters - 1]) {
    const value = weights[index], epsilon = 1e-5;
    weights[index] = value + epsilon; const plus = modelLossAndGradient(weights, data, sample, model).loss;
    weights[index] = value - epsilon; const minus = modelLossAndGradient(weights, data, sample, model).loss;
    weights[index] = value;
    assert.ok(Math.abs((plus - minus) / (2 * epsilon) - gradient[index]) < 2e-7, `parameter ${index}`);
  }
});
