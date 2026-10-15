import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {root} from '../project.ts';
import {MODEL_IDS} from '../dist/types.js';
import {modelInfo, modelPresets, optimizerProfile} from '../dist/models.js';

interface Run {seed: number; firstTest99: number | null}
interface Profile {learningRate: number; weightDecay: number; runs: Run[]}
interface Plan {widths: number[]; learningRates: number[]; weightDecays: number[]}
interface Summary {plan: {confirmationSeeds: number[]; confirmationEpochs: number}; results: {width: number; winner: Profile | null; profiles: Profile[]}[]}
interface Ranked {learningRate: number; weightDecay: number; seeds: number[]; firstTest99: number[]; plateauEpochs: number[]}
interface Late {width: number; seeds: number[]; epochs: number; latest: Ranked | null; longestPlateau: Ranked | null; longestDelayedPlateau: Ranked | null}
const json = async <T>(path: string): Promise<T> => JSON.parse(await readFile(new URL(path, root), 'utf8')) as T;
const late = (width: number) => json<Late>(`../../docs/grokking-results/dense-optimizer-late/w${width}/summary.json`);

test('website optimizer presets and displayed crossings match the archived Rust/CUDA confirmations', async () => {
  // Each archived grid comes with the plan that bounds its rates and decays.
  const grids = await Promise.all([['dense-optimizer-grid', 'div97-dense-optimizer-grid'], ['dense-optimizer-grid-64-256-512', 'div97-dense-optimizer-grid-64-256-512']].map(async ([archive, plan]) =>
    ({summary: await json<Summary>(`../../docs/grokking-results/${archive}/summary.json`), plan: await json<Plan>(`../../experiments/${plan}.json`)})));
  for (const model of MODEL_IDS) {
    const info = modelInfo(model), {summary, plan} = grids.find(({plan}) => plan.widths.includes(info.width))!;
    const result = summary.results.find(result => result.width === info.width)!;
    const selected = result.profiles.find(profile => plan.learningRates.includes(profile.learningRate) && plan.weightDecays.includes(profile.weightDecay))!;
    assert.deepEqual(optimizerProfile(model, 'fast'), optimizerProfile(model));
    assert.equal(info.learningRate, selected.learningRate);
    assert.equal(info.weightDecay, selected.weightDecay);
    assert.deepEqual(info.nativeSeeds, selected.runs.map(run => run.seed));
    assert.deepEqual(info.nativeTest99Epochs, selected.runs.map(run => run.firstTest99));
    assert.equal(info.nativeBudget, summary.plan.confirmationEpochs);
    assert.equal(info.nativePlateauEpochs, undefined);
  }
});

test('slow presets are the latest archived Rust/CUDA configurations that reach 99% on every seed', async () => {
  for (const model of MODEL_IDS) {
    const {width} = modelInfo(model), {latest, seeds, epochs} = await late(width), fast = optimizerProfile(model, 'fast');
    if (!latest) {assert.throws(() => optimizerProfile(model, 'slow')); continue;}
    const profile = optimizerProfile(model, 'slow');
    assert.equal(profile.learningRate, latest.learningRate);
    assert.equal(profile.weightDecay, latest.weightDecay);
    assert.deepEqual(profile.nativeSeeds, latest.seeds);
    assert.deepEqual(profile.nativeTest99Epochs, latest.firstTest99);
    assert.deepEqual(seeds, fast.nativeSeeds);
    assert.equal(profile.nativeBudget, epochs);
    // A slow preset shows its plateau when every seed holds one.
    assert.deepEqual(profile.nativePlateauEpochs, Math.min(...latest.plateauEpochs) >= 25 ? latest.plateauEpochs : undefined);
    assert.ok(Math.min(...latest.firstTest99) > Math.max(...fast.nativeTest99Epochs.map(epoch => epoch ?? Infinity)));
  }
});

test('plateau presets have the longest archived memorization plateau that differs from the slow preset', async () => {
  const plateaus = MODEL_IDS.filter(model => modelPresets(model).includes('plateau'));
  // Widths below 128 never hold a plateau on every seed.
  assert.ok(plateaus.every(model => modelInfo(model).width >= 128));
  for (const model of MODEL_IDS) {
    const {width} = modelInfo(model), {latest, longestPlateau, longestDelayedPlateau, epochs} = await late(width);
    // Where the longest plateau is also the latest crossing, the longest that groks by the website's rule replaces it.
    const same = (a: Ranked | null, b: Ranked | null) => a?.learningRate === b?.learningRate && a?.weightDecay === b?.weightDecay;
    const expected = longestPlateau && !same(longestPlateau, latest) ? longestPlateau : longestDelayedPlateau;
    if (!expected) {assert.throws(() => optimizerProfile(model, 'plateau')); continue;}
    const profile = optimizerProfile(model, 'plateau');
    assert.ok(!same(expected, latest));
    assert.equal(profile.learningRate, expected.learningRate);
    assert.equal(profile.weightDecay, expected.weightDecay);
    assert.deepEqual(profile.nativeSeeds, expected.seeds);
    assert.deepEqual(profile.nativeTest99Epochs, expected.firstTest99);
    assert.deepEqual(profile.nativePlateauEpochs, expected.plateauEpochs);
    assert.equal(profile.nativeBudget, epochs);
    assert.ok(Math.min(...expected.plateauEpochs) >= 25);
  }
});
