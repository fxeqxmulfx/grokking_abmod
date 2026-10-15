// Bitwise comparison behind row-groups-bitwise.log. $S/wasm-before held the
// pre-row-group dist binaries (rows 1); their SHA-256 hashes head the log.
// "dist" compares the rebuilt site binaries for all widths, 1, 2 and 4 threads.
// Bitwise comparison of candidate CPU trainers against the pre-change binaries.
// Usage: node compare-wasm.mjs <rows|dist> [widths=16,32,64,128,256] [threads=2,4]
// "dist" compares the built site binaries instead of in-memory candidates.
import {readFile} from 'node:fs/promises';
const W = '/home/misha/rust_projects/grokking_abmod/web/own-channel/';
const S = '/tmp/claude-1000/-home-misha-rust-projects-grokking-abmod/c59c6907-833b-4928-bb9a-68b7008917ae/scratchpad';
const {assembleCPU} = await import(W + 'build/scripts/assemble-cpu.js');
const {CPU_CONFIG} = await import(W + 'build/scripts/cpu-config.js');
const {WASMTrainer} = await import(W + 'dist/wasm.js');
const {ParallelWASMTrainer} = await import(W + 'dist/parallel-wasm.js');
const {Sampler} = await import(W + 'dist/model.js');
const {initialModelWeights, modelInfo} = await import(W + 'dist/models.js');
const {nodeCPUThread} = await import(W + 'build/tests/node-cpu.js');

const dist = process.argv[2] === 'dist', rows = dist ? null : Number(process.argv[2] ?? 4);
const widths = (process.argv[3] ?? '16,32,64,128,256').split(',').map(Number);
const threadCounts = (process.argv[4] ?? '2,4').split(',').filter(Boolean).map(Number);
const id = width => width === 128 ? 'dense-rotary' : `dense-rotary-${width}`;
// WebAssembly leaves NaN payloads and signs nondeterministic, so every NaN
// compares equal; all other values must match bit for bit.
const bytesOf = array => {
  if (!(array instanceof Float32Array || array instanceof Float64Array)) return Buffer.from(array.buffer, array.byteOffset, array.byteLength);
  const copy = array.slice();
  for (let i = 0; i < copy.length; i++) if (copy[i] !== copy[i]) copy[i] = NaN;
  return Buffer.from(copy.buffer);
};
let checks = 0, nanChecks = 0;
function same(x, y, label) {
  if ((x instanceof Float32Array || x instanceof Float64Array) && x.some(value => value !== value)) nanChecks++;
  const p = bytesOf(x), q = bytesOf(y);
  if (p.length !== q.length) throw new Error(`${label}: length ${p.length} != ${q.length}`);
  if (!p.equals(q)) {
    const a = new Uint32Array(p.buffer, p.byteOffset, p.length / 4), b = new Uint32Array(q.buffer, q.byteOffset, q.length / 4);
    let count = 0, first = -1;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) {count++; if (first < 0) first = i;}
    throw new Error(`${label}: ${count} words differ, first at ${first}`);
  }
  checks++;
}
// Mixed zero inputs: tokens below 30 have a zero first rotary pair, and the
// first two Dense columns are zero. With `infinite`, A[0][5] is +Infinity.
function special(model, infinite) {
  const weights = initialModelWeights(11, model), width = modelInfo(model).width;
  for (let k = 0; k < 30; k++) weights[k * width] = weights[k * width + 1] = 0;
  for (let i = 0; i < width; i++) weights[97 * width + i * width] = weights[97 * width + i * width + 1] = 0;
  return {weights, patch: infinite ? [[97 * width + 5, Infinity]] : []};
}
const sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 37, 48, 512];
async function serial(model, scalar, before, after, {weights, patch = []} = {}, label, finite = true) {
  const options = {model, scalar, learningRate: .001, decay: .1, weights};
  const a = await WASMTrainer.create(11, {...options, bytes: before}), b = await WASMTrainer.create(11, {...options, bytes: after});
  // The constructor accepts only finite weights; write nonfinite ones afterwards.
  for (const t of [a, b]) {for (const [index, value] of patch) t.weights[index] = value; t.exports.prepare();}
  try {
    const sampler = new Sampler(11);
    for (const size of sizes) {
      const sample = sampler.sample().slice(0, size);
      await a.train([sample], {gradientsOnly: true}); await b.train([sample], {gradientsOnly: true});
      same(await a.readGradients(), await b.readGradients(), `${label} gradient ${size}`);
    }
    for (let step = 0; step < 6; step++) {
      const sample = sampler.sample().slice(0, step === 5 ? 48 : 512);
      await a.train([sample]); await b.train([sample]);
      same(await a.readWeights(), await b.readWeights(), `${label} weights step ${step}`);
    }
    const ids = [0, 1, 2, 3, 19, 4655, 4656, 9311];
    const [la, lb] = [await a.readLogits(ids), await b.readLogits(ids)];
    la.forEach((value, i) => same(value, lb[i], `${label} logits ${ids[i]}`));
    a.exports.evaluate(); b.exports.evaluate();
    same(a.metrics, b.metrics, `${label} metrics`);
    for (const [first, end] of [[0, 9312], [5, 9312], [1, 4], [4650, 4663], [9307, 9312]]) {
      a.exports.evaluate_range(first, end); b.exports.evaluate_range(first, end);
      const statistics = t => [new Float64Array(t.exports.memory.buffer, Number(t.exports.statistics.value), 2), new Uint32Array(t.exports.memory.buffer, Number(t.exports.statistics.value) + 16, 2)];
      const [sa, sb] = [statistics(a), statistics(b)];
      same(sa[0], sb[0], `${label} losses ${first}-${end}`); same(sa[1], sb[1], `${label} counts ${first}-${end}`);
    }
    if (finite) for (const t of [a, b]) await t.evaluate();
  } finally {a.destroy(); b.destroy();}
}
async function parallel(model, scalar, before, after, threads, label) {
  const options = {model, scalar, threads, learningRate: .001, decay: .1, workerFactory: nodeCPUThread};
  const a = await ParallelWASMTrainer.create(11, {...options, bytes: before}), b = await ParallelWASMTrainer.create(11, {...options, bytes: after});
  try {
    const sampler = new Sampler(11);
    for (const size of [1, 7, 16, 37, 48, 512]) {
      const sample = sampler.sample().slice(0, size);
      await a.train([sample], {gradientsOnly: true}); await b.train([sample], {gradientsOnly: true});
      same(await a.readGradients(), await b.readGradients(), `${label} gradient ${size}`);
    }
    for (let step = 0; step < 4; step++) {
      const sample = sampler.sample().slice(0, step === 3 ? 48 : 512);
      await a.train([sample]); await b.train([sample]);
    }
    same(await a.readWeights(), await b.readWeights(), `${label} weights`);
    const [ma, mb] = [await a.evaluate(), await b.evaluate()];
    same(Float64Array.from(Object.values(ma)), Float64Array.from(Object.values(mb)), `${label} evaluation`);
  } finally {a.destroy(); b.destroy();}
}
for (const width of widths) for (const variant of ['simd', 'scalar']) {
  const model = id(width), scalar = variant === 'scalar', start = performance.now();
  const config = dist ? CPU_CONFIG[`${Math.max(width, 128)}-${variant}`] : {...CPU_CONFIG[`${Math.max(width, 128)}-${variant}`], rows};
  const before = new Uint8Array(await readFile(`${S}/wasm-before/${model}-${variant}.wasm`));
  const beforeShared = new Uint8Array(await readFile(`${S}/wasm-before/${model}-${variant}-shared.wasm`));
  const after = dist ? new Uint8Array(await readFile(`${W}dist/${model}-${variant}.wasm`)) : await assembleCPU(model, variant, config);
  const afterShared = dist ? new Uint8Array(await readFile(`${W}dist/${model}-${variant}-shared.wasm`)) : await assembleCPU(model, variant, config, true);
  await serial(model, scalar, before, after, {}, `${model}/${variant}`);
  await serial(model, scalar, before, after, special(model, false), `${model}/${variant} zeros`);
  await serial(model, scalar, before, after, special(model, true), `${model}/${variant} infinity`, false);
  for (const threads of threadCounts) await parallel(model, scalar, beforeShared, afterShared, threads, `${model}/${variant}/${threads} threads`);
  console.log(`${model}/${variant} ${JSON.stringify(config)}: identical (${((performance.now() - start) / 1000).toFixed(1)} s)`);
}
console.log(`${checks} bitwise checks passed; ${nanChecks} of them contained NaN.`);
