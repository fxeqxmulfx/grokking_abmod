import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {WASMTrainer} from '../dist/wasm.mjs';
import {Sampler, dataset} from '../dist/model.mjs';
import {modelInfo, initialModelWeights, predictModel, inputMatrix} from '../dist/models.mjs';
import {modelLossAndGradient} from './models-reference.mjs';
import {adamStep} from './reference.mjs';

function close(actual, expected, tolerance, label) {
  let error = 0, index = 0;
  assert.equal(actual.length, expected.length);
  assert.ok(actual.every(Number.isFinite) && expected.every(Number.isFinite), `${label}: nonfinite values`);
  for (let i = 0; i < actual.length; i++) if (Math.abs(actual[i] - expected[i]) > error) { error = Math.abs(actual[i] - expected[i]); index = i; }
  assert.ok(Number.isFinite(error) && error <= tolerance, `${label}: error ${error} at ${index}, tolerance ${tolerance}`);
}

for (const model of ['concat', 'residual', 'neighbor', 'neighbor-concat', 'neighbor-rotary', 'neighbor-rotary-wide']) for (const variant of ['simd', 'scalar']) {
  const bytes = await readFile(new URL(`../dist/${model}-${variant}.wasm`, import.meta.url)), info = modelInfo(model);
  const create = weights => WASMTrainer.create(7, {model, bytes, scalar: variant === 'scalar', weights});
  test(`${model}/${variant} has no imports and matches the seeded initialization`, async () => {
    assert.deepEqual(WebAssembly.Module.imports(await WebAssembly.compile(bytes)), []);
    const engine = await create();
    try { assert.deepEqual(await engine.readWeights(), initialModelWeights(7, model)); assert.ok(engine.info().memoryBytes <= (info.width === 1024 ? 3145728 : 1572864)); }
    finally { engine.destroy(); }
  });
  test(`${model}/${variant} logits and all gradients match f64 backpropagation`, async () => {
    const weights = initialModelWeights(7, model), engine = await create(weights), data = dataset(), projected = inputMatrix(weights, model);
    try {
      const ids = [0, 19, 4655, 4656, 9311], logits = await engine.readLogits(ids);
      ids.forEach((id, i) => close(logits[i], predictModel(weights, data[id * 3], data[id * 3 + 1], model, projected).logits, info.width === 1024 ? 1e-5 : 5e-6, 'logits'));
      for (const n of [1, 37, 48, 512]) {
        const sample = new Sampler().sample().slice(0, n);
        await engine.train([sample], {gradientsOnly: true});
        const actual = await engine.readGradients(), expected = modelLossAndGradient(weights, data, sample, model).gradient;
        if (info.meshDepth) {
          // Square can give large initial operand gradients. Compare each matrix
          // and all sparse edges separately, relative to its reference magnitude.
          const boundary = info.tiedEmbedding ? info.embedding : info.readout;
          for (const [start, end] of [[0, boundary], [boundary, info.gain], [info.gain, info.parameters]].filter(([start, end]) => end > start)) {
            const reference = expected.subarray(start, end), scale = reference.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 1e-4);
            // The concat ablation additionally differentiates a length-128 dense
            // projection; its FP32 accumulation has a larger rounding budget.
            const tolerance = info.projection !== undefined ? 8e-6 : 4e-6;
            close(actual.subarray(start, end), reference, tolerance * scale, `batch ${n}, parameters ${start}..${end}`);
          }
        } else close(actual, expected, 8e-6, `${info.parameters} gradients, batch ${n}`);
      }
    } finally { engine.destroy(); }
  });
  test(`${model}/${variant} independent gains/edges and shared parameter accumulation`, async () => {
    const weights = initialModelWeights(7, model);
    for (let i = info.gain; i < weights.length; i++) weights[i] = .4 + .05 * (i % 11);
    weights[info.gain + 9] = -.5; weights[info.gain + 10] = 0;
    const engine = await create(weights), sample = new Sampler().sample().slice(0, 37);
    try {
      await engine.train([sample], {gradientsOnly: true});
      const actual = await engine.readGradients(), expected = modelLossAndGradient(weights, dataset(), sample, model).gradient;
      if (info.width === 1024) for (const [start, end] of [[0, info.embedding], [info.gain, info.parameters]]) {
        const reference = expected.subarray(start, end), scale = reference.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 1e-4);
        close(actual.subarray(start, end), reference, 4e-6 * scale, 'nontrivial wide weights');
      } else close(actual, expected, 8e-6, 'nontrivial weights');
    } finally { engine.destroy(); }
  });
  test(`${model}/${variant} AdamW matches a single state per unique parameter`, async () => {
    const engine = await create(), expected = Float64Array.from(initialModelWeights(7, model)), moments = new Float64Array(info.parameters * 2), sampler = new Sampler(), data = dataset();
    try {
      for (let step = 1; step <= 3; step++) {
        const sample = sampler.sample().slice();
        adamStep(expected, modelLossAndGradient(expected, data, sample, model).gradient, moments, step);
        await engine.train([sample]); close(await engine.readWeights(), expected, 2e-5, `AdamW step ${step}`);
      }
      const row = await engine.evaluate();
      assert.ok(Number.isFinite(row.trainLoss) && Number.isFinite(row.heldoutLoss));
    } finally { engine.destroy(); }
  });
}

test('concat scalar reference includes all three uses of E and both halves of A', () => {
  const weights = Float64Array.from(initialModelWeights()), sample = new Sampler().sample().slice(0, 3), data = dataset();
  const {gradient} = modelLossAndGradient(weights, data, sample, 'concat');
  for (const index of [0, data[sample[0] * 3] * 64 + 3, 6208 + 19, 6208 + 64 * 64 + 31, 14400 + 5]) {
    const value = weights[index], epsilon = 1e-5;
    weights[index] = value + epsilon; const plus = modelLossAndGradient(weights, data, sample, 'concat').loss;
    weights[index] = value - epsilon; const minus = modelLossAndGradient(weights, data, sample, 'concat').loss;
    weights[index] = value;
    assert.ok(Math.abs((plus - minus) / (2 * epsilon) - gradient[index]) < 2e-8);
  }
});

test('Neighbor seed 7 and seed 11 match the Rust raw checkpoint hashes', () => {
  for (const [seed, hash] of [[7, '586a9cf28bb26ed0a1f58fc67d6b4bf5a2b1918e0a99e32cc0cd2d091cf88ae5'],
                            [11, '0e289f2ea6784e257cdc2acc60fb238d00886d04c402bc34846da009879a43ea']]) {
    const weights = initialModelWeights(seed, 'neighbor');
    assert.equal(weights.length, 38784);
    assert.equal(createHash('sha256').update(new Uint8Array(weights.buffer)).digest('hex'), hash);
  }
});

test('Neighbor f64 derivatives include signed Square inputs, both skips, and the 127→0 edge', () => {
  const weights = Float64Array.from(initialModelWeights(11, 'neighbor')), info = modelInfo('neighbor');
  for (let i = info.gain; i < weights.length; i++) weights[i] = .15 + .6 * Math.sin(i * 1.7);
  const data = dataset(11), sample = [0, 1, 2], {gradient} = modelLossAndGradient(weights, data, sample, 'neighbor');
  for (const index of [data[0] * 128 + 127, info.readout + 127 * 97 + 22,
                       ...Array.from({length: 6}, (_, layer) => info.gain + (layer * 128 + 127) * 2 + 1)]) {
    const value = weights[index], epsilon = 1e-5;
    weights[index] = value + epsilon; const plus = modelLossAndGradient(weights, data, sample, 'neighbor').loss;
    weights[index] = value - epsilon; const minus = modelLossAndGradient(weights, data, sample, 'neighbor').loss;
    weights[index] = value;
    assert.ok(Math.abs((plus - minus) / (2 * epsilon) - gradient[index]) < 2e-7, `parameter ${index}`);
  }
});

test('Rotary roles preserve embedding norms and distinguish swapped nonidentical operands', () => {
  const weights = initialModelWeights(11, 'neighbor-rotary'), info = modelInfo('neighbor-rotary'), input = inputMatrix(weights, info.id);
  for (let k = 0; k < 97; k++) {
    let left = 0, right = 0;
    for (let c = 0; c < 128; c++) { left += input[k * 128 + c] ** 2; right += input[(97 + k) * 128 + c] ** 2; }
    assert.ok(Math.abs(left - right) < 1e-12);
  }
  assert.notDeepEqual(predictModel(weights, 19, 27, info.id, input).rawInput, predictModel(weights, 27, 19, info.id, input).rawInput);
  assert.equal(info.parameters, 13952);
});

test('Rotary checkpoints match Rust byte-for-byte at widths 128 and 1024', () => {
  for (const [model, seed, hash] of [
    ['neighbor-rotary', 7, '1ecf3eb26e7204054441e810f9d0d591e2af53c83797f9980876f6bb709e3ffa'],
    ['neighbor-rotary', 11, '2058f6db2c2f3b7f1ffb46eb588a18f5a2ff1b093e8e8a95ffc6fbe7d883ac0a'],
    ['neighbor-rotary-wide', 7, '3f23d231ce8ab9025aa329a1c2769ac363946dfb40b5b083cbc72d5f90281fa8'],
    ['neighbor-rotary-wide', 11, '55680b38cc7b4db7d12b30b12dbe599489f2e8d86927cc224db832001adf5c49'],
  ]) {
    const weights = initialModelWeights(seed, model);
    assert.equal(createHash('sha256').update(new Uint8Array(weights.buffer)).digest('hex'), hash);
  }
});

test('Rotary f64 derivatives accumulate a, rotated b, and readout when a equals b', () => {
  const info = modelInfo('neighbor-rotary'), weights = Float64Array.from(initialModelWeights(11, info.id)), data = dataset(11);
  for (let i = info.gain; i < weights.length; i++) weights[i] = .15 + .6 * Math.sin(i * 1.7);
  const same = Array.from({length: 4656}, (_, i) => i).find(i => data[i * 3] === data[i * 3 + 1]);
  const sample = [same, 0, 1], residue = data[same * 3], {gradient} = modelLossAndGradient(weights, data, sample, info.id);
  for (const index of [0, 127, residue * 128, residue * 128 + 1, residue * 128 + 126, residue * 128 + 127,
    ...Array.from({length: 6}, (_, layer) => info.gain + (layer * 128 + 127) * 2 + 1)]) {
    const value = weights[index], epsilon = 1e-5;
    weights[index] = value + epsilon; const plus = modelLossAndGradient(weights, data, sample, info.id).loss;
    weights[index] = value - epsilon; const minus = modelLossAndGradient(weights, data, sample, info.id).loss;
    weights[index] = value;
    assert.ok(Math.abs((plus - minus) / (2 * epsilon) - gradient[index]) < 2e-7, `parameter ${index}`);
  }
});
