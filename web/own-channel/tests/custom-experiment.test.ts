import test from 'node:test';
import assert from 'node:assert/strict';
import {modelForWidth, modelId, modelInfo, initialModelWeights, predictModel, trainingOptimizer} from '../dist/models.js';
import {Sampler, dataset} from '../dist/model.js';
import {WASMTrainer} from '../dist/wasm.js';
import {ParallelWASMTrainer} from '../dist/parallel-wasm.js';
import {createEngine} from '../dist/engine.js';
import {cpuMemoryLayout} from '../dist/cpu-memory.js';
import {assembleCPU} from '../scripts/assemble-cpu.ts';
import {CPU_CONFIG} from '../scripts/cpu-config.ts';
import {modelLossAndGradient} from './models-reference.ts';
import {adamStep} from './reference.ts';
import {nodeCPUThread} from './node-cpu.ts';

function close(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance: number, label: string) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) assert.ok(Number.isFinite(actual[i]) && Math.abs(actual[i] - expected[i]) <= tolerance * Math.max(1, Math.abs(expected[i])), `${label}[${i}]: ${actual[i]} vs ${expected[i]}`);
}

test('custom widths validate the Rotate90 pairs and memory limits', () => {
  for (const width of [2, 6, 18, 48, 128, 258, 2048]) {
    const id = modelForWidth(width), info = modelInfo(id);
    assert.equal(info.width, width);
    assert.equal(info.parameters, 97 * width + width * width);
    assert.equal(modelId(id), id);
    assert.ok(cpuMemoryLayout(width, 4).pages <= 8192);
  }
  assert.equal(modelId('dense-rotary-128'), 'dense-rotary');
  for (const width of [0, 1, 3, 17, -2, 2.5, 2050, Infinity, NaN]) assert.throws(() => modelForWidth(width));
  for (const id of ['dense-rotary-0', 'dense-rotary-03', 'dense-rotary-18/../../', 'constructor']) assert.throws(() => modelId(id));
  assert.deepEqual(trainingOptimizer('dense-rotary-18', 'fast', {learningRate: .007, weightDecay: 0}), {learningRate: .007, weightDecay: 0});
});

for (const width of [2, 4, 6, 12, 18, 48, 132, 258]) for (const variant of width % 4 === 0 ? ['scalar', 'simd'] as const : ['scalar'] as const) {
  test(`custom W${width}/${variant} logits and full gradients match independent f64 backpropagation`, async () => {
    const model = modelForWidth(width), bytes = await assembleCPU(model, variant, CPU_CONFIG[`128-${variant}`]);
    const weights = initialModelWeights(11, model), data = dataset(11);
    const engine = await WASMTrainer.create(11, {model, scalar: variant === 'scalar', bytes});
    try {
      const logits = await engine.readLogits([0, 19]);
      [0, 19].forEach((id, i) => close(logits[i], predictModel(weights, data[id * 3], data[id * 3 + 1], model).logits, 8e-6, 'logits'));
      for (const count of [1, 5, 37]) {
        const sample = new Sampler(11).sample().slice(0, count);
        await engine.train([sample], {gradientsOnly: true});
        close(await engine.readGradients(), modelLossAndGradient(weights, data, sample, model).gradient, 2e-5, `batch ${count}`);
      }
      const metrics = await engine.evaluate();
      assert.ok(Number.isFinite(metrics.trainLoss) && Number.isFinite(metrics.heldoutLoss));
    } finally {engine.destroy();}
  });
}

test('custom optimizer reaches the actual engine and AdamW update', async context => {
  const model = modelForWidth(18), optimizer = {learningRate: .007, weightDecay: .123};
  const bytes = await assembleCPU(model, 'scalar', CPU_CONFIG['128-scalar']);
  const fetch = context.mock.method(globalThis, 'fetch', async () => new Response(bytes));
  const engine = await createEngine(11, 'simd', model, 'fast', optimizer) as WASMTrainer;
  try {
    assert.equal(engine.learningRate, optimizer.learningRate); assert.equal(engine.decay, optimizer.weightDecay);
    assert.ok(engine.info().description.includes('scalar'));
    assert.ok(engine.info().note?.includes('multiple of four'));
    const sample = new Sampler(11).sample().slice(0, 5);
    await engine.train([sample], {gradientsOnly: true});
    const expected = Float64Array.from(await engine.readWeights());
    adamStep(expected, Float64Array.from(await engine.readGradients()), new Float64Array(expected.length * 2), 1, {...modelInfo(model), ...optimizer});
    await engine.train([sample]);
    close(await engine.readWeights(), expected, 3e-7, 'custom AdamW');
  } finally {engine.destroy(); fetch.mock.restore();}
});

for (const [width, variant] of [[18, 'scalar'], [48, 'simd']] as const) test(`custom W${width}/${variant} trains with four shared CPU workers`, async () => {
  const model = modelForWidth(width), optimizer = {learningRate: .003, decay: 0};
  const bytes = await assembleCPU(model, variant, CPU_CONFIG[`128-${variant}`], true);
  const engine = await ParallelWASMTrainer.create(11, {model, scalar: variant === 'scalar', bytes, threads: 4, workerFactory: nodeCPUThread, ...optimizer});
  try {
    const sample = new Sampler(11).sample().slice(0, 37);
    await engine.train([sample], {gradientsOnly: true});
    close(await engine.readGradients(), modelLossAndGradient(initialModelWeights(11, model), dataset(11), sample, model).gradient, 2e-5, 'parallel gradients');
    await engine.train([sample]);
    assert.equal(engine.step, 1); assert.equal(engine.info().threads, 4);
    assert.ok(Number.isFinite((await engine.evaluate()).heldoutLoss));
  } finally {engine.destroy();}
});
