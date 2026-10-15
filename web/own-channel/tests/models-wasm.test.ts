import {root} from '../project.ts';
import type {Vector, Weights, ModelId} from '../dist/types.js';
import {MODEL_IDS} from '../dist/types.js';
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

for (const model of MODEL_IDS) for (const variant of ['simd', 'scalar'] as const) {
  const bytes = await readFile(new URL(`dist/${model}-${variant}.wasm`, root)), info = modelInfo(model);
  const create = (weights?: Weights) => WASMTrainer.create(7, {model, bytes: new Uint8Array(bytes), scalar: variant === 'scalar', weights});
  test(`${model}/${variant} has no imports and matches the seeded initialization`, async () => {
    assert.deepEqual(WebAssembly.Module.imports(await WebAssembly.compile(bytes)), []);
    const engine = await create();
    // Weights, gradient and both Adam moments, plus at most 1 MiB for data and activations.
    try { assert.deepEqual(await engine.readWeights(), initialModelWeights(7, model)); assert.ok(engine.info().memoryBytes! <= 16 * info.parameters + 1048576); }
    finally { engine.destroy(); }
  });
  test(`${model}/${variant} logits and all gradients match f64 backpropagation`, async () => {
    const weights = initialModelWeights(7, model), engine = await create(weights), data = dataset(), projected = inputMatrix(weights, model);
    try {
      const ids = [0, 19, 4655, 4656, 9311], logits = await engine.readLogits(ids);
      ids.forEach((id, i) => close(logits[i], predictModel(weights, data[id * 3], data[id * 3 + 1], model, projected).logits, 5e-6, 'logits'));
      for (const n of [1, 37, 48, 512]) {
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
      if (info.width >= 256) for (const [start, end] of [[0, info.embedding], [info.gain, info.parameters]]) {
        const reference = expected.subarray(start, end), scale = reference.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 1e-4);
        // This coherent, non-Kaiming fixture is ill-conditioned at wide widths:
        // f32-level weight noise alone moves the f64 E gradient by 3e-5 at W256.
        // The allowance stays far below any connection, role or sign error.
        close(actual.subarray(start, end), reference, 2e-4 * Math.max(1, scale), 'nontrivial wide weights');
      } else close(actual, expected, 8e-6, 'nontrivial weights');
    } finally { engine.destroy(); }
  });
  test(`${model}/${variant} AdamW matches a single state per unique parameter`, async () => {
    const engine = await create(), expected = Float64Array.from(initialModelWeights(7, model)), moments = new Float64Array(info.parameters * 2), sampler = new Sampler(), data = dataset();
    const suppliedMoments = new Float64Array(info.parameters * 2);
    try {
      for (let step = 1; step <= 3; step++) {
        const sample = sampler.sample().slice();
        await engine.train([sample], {gradientsOnly: true});
        const supplied = Float64Array.from(await engine.readWeights());
        adamStep(supplied, Float64Array.from(await engine.readGradients()), suppliedMoments, step, info);
        adamStep(expected, modelLossAndGradient(expected, data, sample, model).gradient, moments, step, info);
        await engine.train([sample]);
        const actual = await engine.readWeights();
        close(actual, supplied, 2e-7, `AdamW supplied f32 gradient, step ${step}`);
        // End-to-end f64 also includes f32 dot-product rounding. Adam's epsilon
        // amplifies near-zero gradient differences in proportion to the rate.
        close(actual, expected, 2e-5 * Math.max(1, info.learningRate / .001), `f64 model and AdamW step ${step}`);
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
test('Normal-E and Kaiming-A checkpoints match Rust byte-for-byte at all displayed widths', () => {
  for (const [model, hash] of [
    ['dense-rotary-16', '2b063d5498dd54fec02097d17561b06fe977a41c6251730e2a424227e382c3c4'],
    ['dense-rotary-32', 'e81b4ca9d1d0625cbdcde867a6e22bb3ec84140651dcbe1b355ee1c32c36fe39'],
    ['dense-rotary-64', '4433d6019f75d87ae00108d5befb6474a9e0bd4d2df65ce0358677c1b365ff80'],
    ['dense-rotary', 'b90448461caff3b6e04bd9ead0a3f7e7f055e8c1cd71ce19f1069186c74ae04b'],
    ['dense-rotary-256', 'a2b331ff08b75dbb3c397e85b94769f3f4ab973baa9877c24217046652263468'],
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
