import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {root} from '../project.ts';
import {ParallelWASMTrainer} from '../dist/parallel-wasm.js';
import {WASMTrainer} from '../dist/wasm.js';
import {Sampler, dataset} from '../dist/model.js';
import {modelInfo, initialModelWeights} from '../dist/models.js';
import {cpuMemoryLayout} from '../dist/cpu-memory.js';
import type {Vector} from '../dist/types.js';
import {MODEL_IDS} from '../dist/types.js';
import {modelLossAndGradient} from './models-reference.ts';
import {adamStep} from './reference.ts';
import {nodeCPUThread} from './node-cpu.ts';

function close(actual: Vector, expected: Vector, tolerance: number, label: string) {
  assert.equal(actual.length, expected.length); let maximum = 0, at = 0;
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Number.isFinite(actual[i]) && Number.isFinite(expected[i]), `${label}: nonfinite at ${i}`);
    const error = Math.abs(actual[i] - expected[i]); if (error > maximum) {maximum = error; at = i;}
  }
  assert.ok(maximum <= tolerance, `${label}: ${maximum} at ${at}, tolerance ${tolerance}`);
}
test('Shared CPU buffers are aligned, disjoint and bounded', () => {
  for (const width of [16, 32, 128, 1024]) for (const count of [1, 2, 4, 8]) {
    const layout = cpuMemoryLayout(width, count);
    assert.equal(layout.slots.length, count); assert.equal(layout.control % 64, 0);
    for (let slot = 0; slot < count; slot++) {
      const addresses = layout.slots[slot];
      for (const value of Object.values(addresses)) assert.ok(value >= layout.control + count * 64 && value < layout.pages * 65536 && value % 64 === 0);
      assert.ok(addresses.gradient + layout.weightBytes <= addresses.samples);
      assert.equal(addresses.hidden - addresses.input, width * 4);
      if (slot) assert.ok(addresses.gradient >= layout.slots[slot - 1].statistics + 32);
    }
    if (width === 1024 && count === 4) assert.ok(layout.pages * 65536 < 40 * 1024 * 1024);
  }
  assert.throws(() => cpuMemoryLayout(256, 2)); assert.throws(() => cpuMemoryLayout(128, 0));
});

for (const model of MODEL_IDS) for (const variant of ['simd', 'scalar'] as const) for (const threads of [2, 4]) {
  test(`${model}/${variant}/${threads} threads: full gradients, shared AdamW and evaluation`, async () => {
    const bytes = new Uint8Array(await readFile(new URL(`dist/${model}-${variant}-shared.wasm`, root)));
    const serialBytes = new Uint8Array(await readFile(new URL(`dist/${model}-${variant}.wasm`, root)));
    const weights = initialModelWeights(11, model), info = modelInfo(model), data = dataset(11);
    const parallel = await ParallelWASMTrainer.create(11, {model, threads, bytes, scalar: variant === 'scalar', workerFactory: nodeCPUThread});
    const serial = await WASMTrainer.create(11, {model, bytes: serialBytes, scalar: variant === 'scalar'});
    try {
      assert.deepEqual(await parallel.readWeights(), weights);
      assert.deepEqual(await parallel.evaluate(), await serial.evaluate());
      assert.equal(parallel.info().threads, threads);
      const sampler = new Sampler(11);
      for (const size of [1, 7, 37, 48, 512]) {
        const sample = sampler.sample().slice(0, size);
        await parallel.train([sample], {gradientsOnly: true}); await serial.train([sample], {gradientsOnly: true});
        close(await parallel.readGradients(), await serial.readGradients(), 8e-6, `serial gradient batch ${size}`);
        if (size === 7 || size === 48) close(await parallel.readGradients(), modelLossAndGradient(weights, data, sample, model).gradient, 8e-6, `f64 gradient batch ${size}`);
      }
      const expected = Float64Array.from(weights), moments = new Float64Array(info.parameters * 2);
      for (let step = 1; step <= 3; step++) {
        const sample = sampler.sample().slice(0, info.width === 1024 ? 48 : 512);
        adamStep(expected, modelLossAndGradient(expected, data, sample, model).gradient, moments, step);
        await parallel.train([sample]); await serial.train([sample]);
        close(await parallel.readWeights(), expected, 2e-5, `f64 AdamW step ${step}`);
        close(await parallel.readWeights(), await serial.readWeights(), 2e-5, `serial AdamW step ${step}`);
      }
      const row = await parallel.evaluate(), reference = await serial.evaluate();
      assert.equal(row.step, 3); assert.equal(row.epoch, .3);
      assert.ok(Math.abs(row.trainLoss - reference.trainLoss) < 2e-5 && Math.abs(row.heldoutLoss - reference.heldoutLoss) < 2e-5);
      assert.ok(Math.abs(row.trainAccuracy - reference.trainAccuracy) < .001 && Math.abs(row.heldoutAccuracy - reference.heldoutAccuracy) < .001);
      await assert.rejects(parallel.train([[4656]]), /Invalid minibatch/);
      await assert.rejects(parallel.train([[0], [1]], {gradientsOnly: true}), /one gradient batch/);
    } finally {parallel.destroy(); serial.destroy();}
  });
}
