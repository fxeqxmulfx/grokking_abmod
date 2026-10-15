import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {root} from '../project.ts';
import {MODEL_IDS} from '../dist/types.js';
import {modelInfo, modelPresets, optimizerProfile} from '../dist/models.js';
import {pytorchSource} from '../dist/pytorch-export.js';

test('all model and optimizer preset exports contain the selected width, optimizer, seed and budget with standalone uv metadata', async () => {
  const canonical = await readFile(new URL('python/rotate90.py', root), 'utf8');
  for (const model of MODEL_IDS) for (const preset of modelPresets(model)) {
    const source = pytorchSource({model, preset, seed: 0xffffffff, maxEpochs: 17}), profile = optimizerProfile(model, preset);
    if (preset === 'fast') assert.equal(pytorchSource({model, seed: 0xffffffff, maxEpochs: 17}), source);
    assert.ok(source.startsWith('# /// script\n'));
    assert.ok(source.includes('https://download.pytorch.org/whl/cpu'));
    assert.ok(source.includes(`DEFAULT_WIDTH = ${modelInfo(model).width}\n`));
    assert.ok(source.includes('DEFAULT_SEED = 4294967295\n'));
    assert.ok(source.includes('DEFAULT_EPOCHS = 17\n'));
    assert.ok(source.includes(`DEFAULT_LEARNING_RATE = ${profile.learningRate}\n`));
    assert.ok(source.includes(`DEFAULT_WEIGHT_DECAY = ${profile.weightDecay}\n`));
    assert.equal(source.replace(/^DEFAULT_(WIDTH|SEED|EPOCHS|LEARNING_RATE|WEIGHT_DECAY) = [\d.]+$/gm, ''), canonical.replace(/^DEFAULT_(WIDTH|SEED|EPOCHS|LEARNING_RATE|WEIGHT_DECAY) = [\d.]+$/gm, ''));
    assert.ok(!/torch\.save|pickle|base64/.test(source));
  }
  for (const seed of [-1, .5, NaN, 0x100000000]) assert.throws(() => pytorchSource({model: 'dense-rotary', seed, maxEpochs: 1}));
  for (const maxEpochs of [0, 1.5, 10001]) assert.throws(() => pytorchSource({model: 'dense-rotary', seed: 0, maxEpochs}));
  assert.throws(() => pytorchSource({model: 'dense-rotary-16', preset: 'plateau', seed: 0, maxEpochs: 1}));
});
