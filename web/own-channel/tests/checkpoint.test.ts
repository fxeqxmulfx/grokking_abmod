import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {root} from '../project.ts';
import {WASMTrainer} from '../dist/wasm.js';
import {Sampler} from '../dist/model.js';
import type {ModelId} from '../dist/types.js';

interface Fixture {seed: number; steps: number; models: {model: ModelId; variant: 'scalar' | 'simd'; weights: string; gradients: string; logits: string; wasm: string}[]}
const fixture = JSON.parse(await readFile(new URL('tests/fixtures/cpu-before-optimization.json', root), 'utf8')) as Fixture;
const hash = (bytes: Uint8Array<ArrayBuffer> | string) => createHash('sha256').update(bytes).digest('hex');
for (const expected of fixture.models) test(`${expected.model}/${expected.variant}: preserve pre-optimization f32 checkpoints bit-for-bit`, async () => {
  const bytes = new Uint8Array(await readFile(new URL(`dist/${expected.model}-${expected.variant}.wasm`, root)));
  // Historical fixtures retain their original optimizer even when tab presets change.
  const engine = await WASMTrainer.create(fixture.seed, {model: expected.model, scalar: expected.variant === 'scalar', bytes, learningRate: .001, decay: .1});
  try {
    const sample = new Sampler(fixture.seed).sample().slice();
    for (let i = 0; i < fixture.steps; i++) await engine.train([sample]);
    assert.equal(hash(new Uint8Array((await engine.readWeights()).buffer)), expected.weights, 'Every trained parameter must match the original CPU implementation.');
    assert.equal(hash(new Uint8Array((await engine.readGradients()).buffer)), expected.gradients, 'Every accumulated gradient must match.');
    assert.equal(hash(JSON.stringify((await engine.readLogits()).map(value => Array.from(value)))), expected.logits, 'Probe logits must match.');
  } finally {engine.destroy();}
});
