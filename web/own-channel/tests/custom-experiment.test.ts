import test from 'node:test';
import assert from 'node:assert/strict';
import {modelForWidth, modelId, modelInfo, initialModelWeights, trainingOptimizer} from '../dist/models.js';
import {Sampler, dataset} from '../dist/model.js';
import {WASMTrainer} from '../dist/wasm.js';
import {ParallelWASMTrainer} from '../dist/parallel-wasm.js';
import {createEngine} from '../dist/engine.js';
import {cpuMemoryLayout} from '../dist/cpu-memory.js';
import {readFile} from 'node:fs/promises';
import {root} from '../project.ts';
import {modelLossAndGradient} from './models-reference.ts';
import {adamStep} from './reference.ts';
import {nodeCPUThread} from './node-cpu.ts';

function close(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance: number, label: string) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) assert.ok(Number.isFinite(actual[i]) && Math.abs(actual[i] - expected[i]) <= tolerance * Math.max(1, Math.abs(expected[i])), `${label}[${i}]: ${actual[i]} vs ${expected[i]}`);
}

test('custom settings accept exactly the five prebuilt widths', () => {
  for (const width of [16, 32, 64, 128, 256]) {
    const id = modelForWidth(width), info = modelInfo(id);
    assert.equal(info.width, width);
    assert.equal(info.parameters, 97 * width + width * width);
    assert.equal(modelId(id), id);
    assert.ok(cpuMemoryLayout(width, 4).pages <= 8192);
  }
  for (const width of [0, 1, 3, 17, 18, 48, 512, -2, 2.5, 2050, Infinity, NaN]) assert.throws(() => modelForWidth(width));
  for (const id of ['dense-rotary-0', 'dense-rotary-03', 'dense-rotary-18/../../', 'constructor']) assert.throws(() => modelId(id));
  assert.deepEqual(trainingOptimizer('dense-rotary-32', 'fast', {learningRate: .007, weightDecay: 0}), {learningRate: .007, weightDecay: 0});
});

test('custom optimizer reaches the actual engine and AdamW update', async context => {
  const model = modelForWidth(32), optimizer = {learningRate: .007, weightDecay: .123};
  const bytes = new Uint8Array(await readFile(new URL(`dist/${model}-scalar.wasm`, root)));
  const fetch = context.mock.method(globalThis, 'fetch', async () => new Response(bytes));
  const engine = await createEngine(11, 'scalar', model, 'fast', optimizer) as WASMTrainer;
  try {
    assert.equal(engine.learningRate, optimizer.learningRate); assert.equal(engine.decay, optimizer.weightDecay);
    assert.ok(engine.info().description.includes('scalar'));
    const sample = new Sampler(11).sample().slice(0, 5);
    await engine.train([sample], {gradientsOnly: true});
    const expected = Float64Array.from(await engine.readWeights());
    adamStep(expected, Float64Array.from(await engine.readGradients()), new Float64Array(expected.length * 2), 1, {...modelInfo(model), ...optimizer});
    await engine.train([sample]);
    close(await engine.readWeights(), expected, 3e-7, 'custom AdamW');
  } finally {engine.destroy(); fetch.mock.restore();}
});

for (const [width, variant] of [[32, 'scalar'], [64, 'simd']] as const) test(`custom W${width}/${variant} trains with four shared CPU workers`, async () => {
  const model = modelForWidth(width), optimizer = {learningRate: .003, decay: 0};
  const bytes = new Uint8Array(await readFile(new URL(`dist/${model}-${variant}-shared.wasm`, root)));
  const engine = await ParallelWASMTrainer.create(11, {model, scalar: variant === 'scalar', bytes, threads: 4, workerFactory: nodeCPUThread, ...optimizer});
  try {
    const sample = new Sampler(11).sample().slice(0, 37);
    await engine.train([sample], {gradientsOnly: true});
    close(await engine.readGradients(), modelLossAndGradient(initialModelWeights(11, model), dataset(11), sample, model).gradient, 2e-5, 'parallel gradients');
    assert.equal(engine.engine.learningRate, optimizer.learningRate); assert.equal(engine.engine.decay, 0);
    await engine.train([sample]);
    assert.equal(engine.step, 1); assert.equal(engine.info().threads, 4);
    assert.ok(Number.isFinite((await engine.evaluate()).heldoutLoss));
  } finally {engine.destroy();}
});
