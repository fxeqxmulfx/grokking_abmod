import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {cpus} from 'node:os';
import {createHash} from 'node:crypto';
import {root} from '../project.ts';
import {WASMTrainer} from '../dist/wasm.js';
import {ParallelWASMTrainer} from '../dist/parallel-wasm.js';
import {Sampler} from '../dist/model.js';
import {nodeCPUThread} from '../tests/node-cpu.ts';

const directory = process.argv[2] ?? '/tmp/grokking-cpu-benchmark';
const repetitions = Number(process.env.GROKKING_BENCH_REPETITIONS ?? 5);
if (!Number.isInteger(repetitions) || repetitions < 3 || repetitions > 31) throw new Error('Benchmark repetitions must be 3..31.');
const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
await mkdir(directory, {recursive: true});
const results = [];
for (const model of ['dense-rotary', 'dense-rotary-256'] as const) for (const variant of ['simd', 'scalar'] as const) for (const threads of [1, 2, 4]) {
  const bytes = new Uint8Array(await readFile(new URL(`dist/${model}-${variant}${threads === 1 ? '' : '-shared'}.wasm`, root)));
  const options = {model, scalar: variant === 'scalar', bytes};
  const engine = threads === 1 ? await WASMTrainer.create(11, options) : await ParallelWASMTrainer.create(11, {...options, threads, workerFactory: nodeCPUThread});
  try {
    const sampler = new Sampler(11), sample = sampler.sample().slice();
    for (let i = 0; i < 3; i++) await engine.train([sample]);
    const trainMs: number[] = [], evaluateMs: number[] = [];
    for (let i = 0; i < repetitions; i++) {const start = performance.now(); await engine.train([sample]); trainMs.push(performance.now() - start);}
    // Warm every worker's forward/evaluation path before timing it.
    await engine.evaluate();
    for (let i = 0; i < Math.max(3, Math.floor(repetitions / 2)); i++) {const start = performance.now(); await engine.evaluate(); evaluateMs.push(performance.now() - start);}
    const weights = await engine.readWeights();
    const result = {model, variant, threads, batch: sample.length, trainMs, evaluateMs, trainMedianMs: median(trainMs), evaluateMedianMs: median(evaluateMs), approximateEpochMs: 10 * median(trainMs) + median(evaluateMs), memoryBytes: engine.info().memoryBytes, wasmSha256: createHash('sha256').update(bytes).digest('hex'), weightsSha256: createHash('sha256').update(new Uint8Array(weights.buffer)).digest('hex')};
    results.push(result); console.log(JSON.stringify(result));
    // Preserve partial measurements if the long run is interrupted.
    await writeFile(`${directory}/report.json`, JSON.stringify({node: process.version, cpu: cpus()[0]?.model, logicalCPUs: cpus().length, seed: 11, repetitions, note: 'Models and engines measured sequentially; epoch estimate uses ten full batches and full-split evaluation, so slightly overestimates the 48-row final batch.', results}, null, 2) + '\n');
  } finally {engine.destroy();}
}
