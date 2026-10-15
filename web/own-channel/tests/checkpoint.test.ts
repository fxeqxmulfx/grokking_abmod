import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {root} from '../project.ts';
import {WASMTrainer} from '../dist/wasm.js';
import {ParallelWASMTrainer} from '../dist/parallel-wasm.js';
import {nodeCPUThread} from './node-cpu.ts';
import {Sampler} from '../dist/model.js';
import type {ModelId} from '../dist/types.js';
import {MODEL_IDS} from '../dist/types.js';

interface Fixture {seed: number; steps: number; models: {model: string; variant: 'scalar' | 'simd'; weights: string; gradients: string; logits: string; wasm: string}[]}
const fixture = JSON.parse(await readFile(new URL('tests/fixtures/cpu-before-optimization.json', root), 'utf8')) as Fixture;
const hash = (bytes: Uint8Array<ArrayBuffer> | string) => createHash('sha256').update(bytes).digest('hex');
// W1024 left the website; its historical checkpoints stay in the fixture.
for (const expected of fixture.models.filter(({model}) => (MODEL_IDS as readonly string[]).includes(model))) test(`${expected.model}/${expected.variant}: preserve pre-optimization f32 checkpoints bit-for-bit`, async () => {
  const bytes = new Uint8Array(await readFile(new URL(`dist/${expected.model}-${expected.variant}.wasm`, root)));
  // Historical fixtures retain their original optimizer even when tab presets change.
  const engine = await WASMTrainer.create(fixture.seed, {model: expected.model as ModelId, scalar: expected.variant === 'scalar', bytes, learningRate: .001, decay: .1});
  try {
    const sample = new Sampler(fixture.seed).sample().slice();
    for (let i = 0; i < fixture.steps; i++) await engine.train([sample]);
    assert.equal(hash(new Uint8Array((await engine.readWeights()).buffer)), expected.weights, 'Every trained parameter must match the original CPU implementation.');
    assert.equal(hash(new Uint8Array((await engine.readGradients()).buffer)), expected.gradients, 'Every accumulated gradient must match.');
    assert.equal(hash(JSON.stringify((await engine.readLogits()).map(value => Array.from(value)))), expected.logits, 'Probe logits must match.');
  } finally {engine.destroy();}
});

for (const [model, variant, threads] of [['dense-rotary-32', 'scalar', 1], ['dense-rotary', 'simd', 1], ['dense-rotary-64', 'simd', 4], ['dense-rotary', 'scalar', 2]] as const) {
  test(`${model}/${variant}/${threads}: resume AdamW exactly after releasing the engine`, async () => {
    const bytes = new Uint8Array(await readFile(new URL(`dist/${model}-${variant}${threads > 1 ? '-shared' : ''}.wasm`, root)));
    const create = () => threads > 1
      ? ParallelWASMTrainer.create(11, {model, bytes, scalar: variant === 'scalar', threads, workerFactory: nodeCPUThread})
      : WASMTrainer.create(11, {model, bytes, scalar: variant === 'scalar'});
    const original = await create(), sampler = new Sampler(11);
    let saved, expected;
    try {
      for (let i = 0; i < 3; i++) await original.train([sampler.sample().slice(0, 37)]);
      saved = structuredClone(original.checkpoint());
      const next = sampler.sample().slice(0, 37);
      await original.train([next]); expected = {checkpoint: original.checkpoint(), sample: next};
    } finally {original.destroy();}
    const resumed = await create();
    try {
      resumed.restore(saved);
      await resumed.train([expected.sample]);
      assert.deepEqual(resumed.checkpoint(), expected.checkpoint, 'Weights, both AdamW moments, step and bias corrections must match an uninterrupted run bit-for-bit.');
      assert.throws(() => resumed.restore({...saved, step: -1}), /Invalid training checkpoint/);
      assert.throws(() => resumed.restore({...saved, weights: new Float32Array(1)}), /Invalid training checkpoint/);
      assert.deepEqual(resumed.checkpoint(), expected.checkpoint, 'Invalid checkpoints must not change the engine.');
    } finally {resumed.destroy();}
  });
}
