import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {Random, initialWeights, dataset, Sampler, packBatch, TRAIN, P, GAIN, Grokking, ShortGrokking} from '../dist/model.mjs';
import {lossAndGradient} from './reference.mjs';

test('u32 SplitMix64 matches an independent BigInt implementation', () => {
  for (const seed of [0n, 7n, 0xffffffffffffffffn, 0x123456789abcdef0n]) {
    let state = seed; const random = new Random(seed), mask = (1n << 64n) - 1n;
    for (let i = 0; i < 2000; i++) {
      state = (state + 0x9e3779b97f4a7c15n) & mask;
      let value = state;
      value = ((value ^ value >> 30n) * 0xbf58476d1ce4e5b9n) & mask;
      value = ((value ^ value >> 27n) * 0x94d049bb133111ebn) & mask;
      value ^= value >> 31n;
      assert.equal(random.next(), Number(value & 0xffffffffn)); assert.equal(random.resultHigh, Number(value >> 32n));
    }
  }
});

test('seed-7 initialization is byte-identical to the original Rust depth-5 run', () => {
  const weights = initialWeights();
  assert.equal(weights.length, 18880);
  assert.equal(createHash('sha256').update(new Uint8Array(weights.buffer)).digest('hex'), '35e8f8f14779106b3b66d4b92c0ac790a3741dbf8b6f8b3dd8e9f50d6a66e826');
});

test('split is complete and division targets are correct; first rows match the Rust artifact', () => {
  const data = dataset(); const pairs = new Set();
  assert.deepEqual(Array.from(data.slice(0, 18)), [7, 92, 18, 65, 58, 63, 6, 4, 50, 9, 43, 16, 34, 1, 34, 35, 87, 45]);
  for (let i = 0; i < data.length; i += 3) {
    const [a, b, target] = data.slice(i, i + 3); assert.equal(target * b % P, a); pairs.add(`${a},${b}`);
  }
  assert.equal(pairs.size, 9312); assert.equal(data.length, 3 * 9312);
});

test('shuffled epochs contain each example once, with an exact 48-row tail', () => {
  const sampler = new Sampler();
  for (let epoch = 0; epoch < 3; epoch++) {
    const samples = Array.from({length: 10}, () => sampler.sample().slice());
    assert.deepEqual(samples.map(s => s.length), [...Array(9).fill(512), 48]);
    assert.equal(new Set(samples.flatMap(s => Array.from(s))).size, TRAIN);
  }
});

test('sparse input lists contain exactly the corresponding minibatch rows', () => {
  const data = dataset();
  for (const n of [1, 37, 48, 512]) {
    const sample = new Sampler().sample().slice(0, n), buffer = new Uint32Array(2000);
    const packed = packBatch(data, sample, buffer, 7);
    assert.equal(packed.end, 7 + 3 * n + 2 * P + 1);
    for (let bin = 0; bin < 2 * P; bin++) {
      const expected = Array.from(sample, (id, row) => [id, row]).filter(([id]) => data[id * 3 + Number(bin >= P)] === bin % P).map(([, row]) => row);
      assert.deepEqual(Array.from(buffer.slice(buffer[packed.bins + bin], buffer[packed.bins + bin + 1])), expected);
    }
  }
});

test('independent reference gradients match finite differences, including nonpositive gains', () => {
  const weights = Float64Array.from(initialWeights()), data = dataset(), samples = [0, 1, 2, 3, 4, 5, 6];
  weights[GAIN + 9] = -.5; weights[GAIN + 64 + 10] = 0; weights[GAIN + 128 + 11] = 1.4;
  const {gradient} = lossAndGradient(weights, data, samples);
  const candidates = [data[0] * 64 + 3, 12416 + 3 * P + 22, GAIN + 11, GAIN + 128 + 11, GAIN + 9];
  for (const index of candidates) {
    const original = weights[index], epsilon = 1e-5;
    weights[index] = original + epsilon; const plus = lossAndGradient(weights, data, samples).loss;
    weights[index] = original - epsilon; const minus = lossAndGradient(weights, data, samples).loss;
    weights[index] = original; assert.ok(Math.abs((plus - minus) / (2 * epsilon) - gradient[index]) < 1e-8);
  }
  for (let layer = 0; layer < 4; layer++) assert.equal(gradient[GAIN + layer * 64 + 9], 0);
});

test('grokking requires a confirmed earlier plateau and the first held-out target event', () => {
  const criterion = new Grokking();
  for (let i = 0; i < 4; i++) criterion.observe({epoch: i * 25, trainAccuracy: 1, heldoutAccuracy: .05});
  assert.equal(criterion.plateau, false);
  criterion.observe({epoch: 100, trainAccuracy: 1, heldoutAccuracy: .05}); assert.equal(criterion.plateau, true);
  assert.equal(criterion.observe({epoch: 2000, trainAccuracy: 1, heldoutAccuracy: .99}), 'Delayed grokking');
  const direct = new Grokking(); assert.equal(direct.observe({epoch: 300, trainAccuracy: 1, heldoutAccuracy: .99}), 'Generalized');
  for (let i = 0; i < 5; i++) direct.observe({epoch: 400 + i * 25, trainAccuracy: 1, heldoutAccuracy: .05});
  assert.equal(direct.observe({epoch: 1000, trainAccuracy: 1, heldoutAccuracy: .99}), 'Generalized');
});

test('later qualifying streaks preserve the first confirmed plateau timestamps', () => {
  const criterion = new Grokking();
  for (let i = 0; i < 5; i++) criterion.observe({epoch: 150 + i * 25, trainAccuracy: 1, heldoutAccuracy: .02});
  criterion.observe({epoch: 275, trainAccuracy: .98, heldoutAccuracy: .02});
  for (let i = 0; i < 6; i++) criterion.observe({epoch: 4000 + i * 25, trainAccuracy: 1, heldoutAccuracy: .05});
  assert.equal(criterion.onset, 150);
  assert.equal(criterion.confirmed, 250);
  assert.equal(criterion.observe({epoch: 5350, trainAccuracy: 1, heldoutAccuracy: .99}), 'Delayed grokking');
});

test('short grokking requires 25 continuous epochs before a fast first 90% crossing', () => {
  const criterion = new ShortGrokking();
  for (let epoch = 100; epoch <= 124; epoch++) criterion.observe({epoch, trainAccuracy: 1, heldoutAccuracy: .05});
  assert.equal(criterion.plateau.epochs, 24);
  criterion.observe({epoch:125, trainAccuracy:1, heldoutAccuracy:.05});
  criterion.observe({epoch:350, trainAccuracy:1, heldoutAccuracy:.9});
  assert.equal(criterion.observe({epoch:400, trainAccuracy:1, heldoutAccuracy:.99}), 'Short-plateau grokking');
  assert.equal(criterion.perfect.epochs, 25); assert.equal(criterion.rise, 225);
});

test('train dips, missing observations, and later accuracy dips cannot qualify short grokking', () => {
  for (const dip of ['train','missing']) {
    const criterion = new ShortGrokking();
    for (let epoch = 0; epoch <= 40; epoch++) {
      if (dip === 'missing' && epoch === 20) continue;
      criterion.observe({epoch, trainAccuracy: dip === 'train' && epoch === 20 ? .98 : 1, heldoutAccuracy:.05});
    }
    assert.equal(criterion.observe({epoch:100, trainAccuracy:1, heldoutAccuracy:.99}), 'Generalized');
  }
  const slow = new ShortGrokking();
  for (let epoch = 0; epoch <= 25; epoch++) slow.observe({epoch, trainAccuracy:1, heldoutAccuracy:.05});
  slow.observe({epoch:300, trainAccuracy:1, heldoutAccuracy:.9});
  slow.observe({epoch:350, trainAccuracy:1, heldoutAccuracy:.05});
  assert.equal(slow.observe({epoch:351, trainAccuracy:1, heldoutAccuracy:.99}), 'Generalized');
  assert.equal(slow.rise, 275); assert.equal(slow.first90, 300);
});
