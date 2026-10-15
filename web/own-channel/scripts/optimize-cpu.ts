import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {cpus} from 'node:os';
import {createHash} from 'node:crypto';
import {root} from '../project.ts';
import {CPU_CONFIG} from './cpu-config.ts';
import type {CPUKernelConfig} from './cpu-config.ts';
import {assembleCPU} from './assemble-cpu.ts';
import {WASMTrainer} from '../dist/wasm.js';
import {Sampler} from '../dist/model.js';
import type {ModelId} from '../dist/types.js';

// A bounded local hill-climb. Build candidates in memory, verify their actual
// outputs, measure them serially, and confirm improvements before promotion.
const arguments_ = process.argv.slice(2);
function argument(name: string, fallback: string) {const at = arguments_.indexOf(name); return at < 0 ? fallback : arguments_[at + 1];}
const directory = argument('--output', '/tmp/grokking-cpu-optimization');
const rounds = Number(argument('--rounds', '2')), apply = arguments_.includes('--apply'), widths = argument('--widths', '128,256').split(',').map(Number);
if (!Number.isInteger(rounds) || rounds < 1 || rounds > 1000) throw new Error('Use --rounds 1..1000 [--widths 128,256] [--apply] [--output directory].');
const config = structuredClone(CPU_CONFIG);
const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const hash = (bytes: Uint8Array<ArrayBuffer>) => createHash('sha256').update(bytes).digest('hex');
type Profile = keyof typeof CPU_CONFIG;
interface Timing {trainMs: number[]; evaluateMs: number[]; train: number; evaluate: number; epoch: number}
interface Trial {round: number; profile: Profile; config: CPUKernelConfig; sha256?: string; correct: boolean; timing?: Timing; error?: string}
const trials: Trial[] = [], promotions: {round: number; profile: Profile; previous: CPUKernelConfig; next: CPUKernelConfig; reference: Timing; candidate: Timing}[] = [];
const sourceHashes: Record<string, string> = {};
for (const source of ['wasm/dense-rotary.wat', 'wasm/simd.wat', 'wasm/scalar.wat', 'wasm/rotary-simd.wat', 'wasm/rotary-scalar.wat', 'scripts/cpu-kernels.ts']) sourceHashes[source] = hash(new Uint8Array(await readFile(new URL(source, root))));
await mkdir(directory, {recursive: true});
const save = async () => writeFile(`${directory}/report.json`, JSON.stringify({node: process.version, cpu: cpus()[0]?.model, seed: 11, rounds, apply, sourceHashes, config, trials, promotions}, null, 2) + '\n');
function neighbors(current: CPUKernelConfig, simd: boolean, width: number): CPUKernelConfig[] {
  const result = [
    ...[16, 32, 64].map(block => ({...current, block})),
    ...(simd ? [2, 4, 8] : [4, 8, 16]).map(registers => ({...current, registers})),
    ...([2, 4] as const).map(unroll => ({...current, unroll})),
    ...([1, 2, 4] as const).map(rows => ({...current, rows})),
    ...[null, ...(width > 128 ? [width] : []), 0].map(rowThreshold => ({...current, rowThreshold})),
  ];
  const seen = new Set([JSON.stringify(current)]);
  return result.filter(value => {const key = JSON.stringify(value); if (seen.has(key)) return false; seen.add(key); return true;});
}
async function verify(model: ModelId, scalar: boolean, referenceBytes: Uint8Array<ArrayBuffer>, candidateBytes: Uint8Array<ArrayBuffer>) {
  const reference = await WASMTrainer.create(11, {model, scalar, bytes: referenceBytes});
  const candidate = await WASMTrainer.create(11, {model, scalar, bytes: candidateBytes});
  try {
    assert.deepEqual(await candidate.readWeights(), await reference.readWeights());
    const sampler = new Sampler(11);
    for (const size of [1, 37, 48, 512]) {
      const sample = sampler.sample().slice(0, size);
      await reference.train([sample], {gradientsOnly: true}); await candidate.train([sample], {gradientsOnly: true});
      assert.deepEqual(await candidate.readGradients(), await reference.readGradients(), `Full gradient mismatch, batch ${size}.`);
    }
    for (let step = 0; step < 5; step++) {
      const sample = sampler.sample().slice(); await reference.train([sample]); await candidate.train([sample]);
    }
    assert.deepEqual(await candidate.readWeights(), await reference.readWeights(), 'AdamW checkpoint mismatch.');
    assert.deepEqual(await candidate.readLogits(), await reference.readLogits(), 'Logit mismatch.');
    assert.deepEqual(await candidate.evaluate(), await reference.evaluate(), 'Complete-split evaluation mismatch.');
  } finally {reference.destroy(); candidate.destroy();}
}
async function measure(model: ModelId, scalar: boolean, bytes: Uint8Array<ArrayBuffer>): Promise<Timing> {
  const engine = await WASMTrainer.create(11, {model, scalar, bytes}), sample = new Sampler(11).sample().slice();
  try {
    for (let i = 0; i < 3; i++) await engine.train([sample]);
    const trainMs: number[] = [], evaluateMs: number[] = [];
    for (let i = 0; i < 5; i++) {const start = performance.now(); await engine.train([sample]); trainMs.push(performance.now() - start);}
    await engine.evaluate();
    for (let i = 0; i < 3; i++) {const start = performance.now(); await engine.evaluate(); evaluateMs.push(performance.now() - start);}
    const train = median(trainMs), evaluate = median(evaluateMs);
    return {trainMs, evaluateMs, train, evaluate, epoch: 10 * train + evaluate};
  } finally {engine.destroy();}
}
function improves(candidate: Timing, reference: Timing) {
  // Require an epoch gain of 3%, and reject regressions in either operation
  // beyond the 3% noise allowance. Confirm again before changing the build.
  return candidate.epoch < reference.epoch * .97 && candidate.train <= reference.train * 1.03 && candidate.evaluate <= reference.evaluate * 1.03;
}
async function publishConfig() {
  const source = `// Generated by scripts/optimize-cpu.ts after exact validation and confirmed benchmarks.\nexport interface CPUKernelConfig {registers: number; block: number; rowThreshold: number | null; unroll: 2 | 4; rows: 1 | 2 | 4}\nexport const CPU_CONFIG: Record<'128-simd' | '128-scalar' | '256-simd' | '256-scalar', CPUKernelConfig> = ${JSON.stringify(config, null, 2)};\n`;
  await writeFile(new URL('scripts/cpu-config.ts', root), source);
}
for (let round = 1; round <= rounds; round++) {
  let changed = false;
  for (const [model, width] of ([['dense-rotary', 128], ['dense-rotary-256', 256]] as const).filter(([, width]) => widths.includes(width))) for (const variant of ['simd', 'scalar'] as const) {
    const profile = `${width}-${variant}` as Profile, previous = {...config[profile]}, scalar = variant === 'scalar';
    const referenceBytes = await assembleCPU(model, variant, previous), reference = await measure(model, scalar, referenceBytes);
    let best = {config: previous, bytes: referenceBytes, timing: reference};
    console.log(JSON.stringify({round, profile, baseline: reference, config: previous}));
    for (const next of neighbors(previous, !scalar, width)) {
      const trial: Trial = {round, profile, config: next, correct: false};
      try {
        const bytes = await assembleCPU(model, variant, next); trial.sha256 = hash(bytes);
        await verify(model, scalar, referenceBytes, bytes); trial.correct = true;
        trial.timing = await measure(model, scalar, bytes);
        if (improves(trial.timing, reference) && trial.timing.epoch < best.timing.epoch) best = {config: next, bytes, timing: trial.timing};
      } catch (error) {
        trial.error = error instanceof Error ? error.message.slice(0, 1000) : String(error);
        // Compilation/numerical failures are rejected; other candidates continue.
      }
      trials.push(trial); await save(); console.log(JSON.stringify(trial));
    }
    if (best.bytes !== referenceBytes) {
      // Alternate order on confirmation to reduce first/last and thermal bias.
      const candidate = await measure(model, scalar, best.bytes), referenceCheck = await measure(model, scalar, referenceBytes);
      if (improves(candidate, referenceCheck)) {
        config[profile] = best.config; changed = true;
        promotions.push({round, profile, previous, next: best.config, reference: referenceCheck, candidate});
        if (apply) await publishConfig();
        await writeFile(`${directory}/${profile}-winner.wasm`, best.bytes);
        console.log(JSON.stringify({promoted: profile, config: best.config, gain: referenceCheck.epoch / candidate.epoch}));
      }
    }
    await save();
  }
  if (!changed) {console.log(`Round ${round}: no confirmed improvement; optimization converged.`); break;}
}
await save(); console.log(`Optimization report: ${directory}/report.json`);
