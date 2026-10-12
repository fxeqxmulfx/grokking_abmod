import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {WASMTrainer} from '../dist/wasm.mjs';
import {Sampler, Grokking, ShortGrokking} from '../dist/model.mjs';
import {modelInfo} from '../dist/models.mjs';

const model = process.argv[2] || 'concat', variant = process.argv[3] || 'simd', directory = process.argv[4];
if (!directory) throw new Error('Usage: node scripts/measure-wasm.mjs <model> <simd|scalar> <output-directory>');
const bytes = await readFile(new URL(`../dist/${model}-${variant}.wasm`, import.meta.url));
const info = modelInfo(model), seed = Number(process.argv[5] ?? info.seed);
const engine = await WASMTrainer.create(seed, {model, bytes, scalar: variant === 'scalar'}), sampler = new Sampler(seed), criterion = info.plateauEpochs ? new ShortGrokking() : new Grokking(), history = [];
const start = performance.now();
try {
  for (;;) {
    const row = await engine.evaluate(); row.phase = criterion.observe(row); history.push(row);
    if (engine.step % 2500 === 0 || row.heldoutAccuracy >= .99) console.log(JSON.stringify(row));
    if (engine.step >= 100000 || row.heldoutAccuracy >= .99) break;
    for (let n = 0; n < info.reportEvery; n += 10) await engine.train(Array.from({length: 10}, () => sampler.sample().slice()));
  }
  await mkdir(directory, {recursive: true});
  await writeFile(`${directory}/history.json`, JSON.stringify(history, null, 2) + '\n');
  const metadata = {model, variant, seed, reportEvery:info.reportEvery, elapsedSeconds: (performance.now() - start) / 1000, wasmSha256: createHash('sha256').update(bytes).digest('hex'), node: process.version, criterion, sourceSha256:{}};
  for (const name of ['wasm.mjs','models.mjs','model.mjs']) metadata.sourceSha256[name] = createHash('sha256').update(await readFile(new URL(`../dist/${name}`,import.meta.url))).digest('hex');
  await writeFile(`${directory}/run.json`, JSON.stringify(metadata, null, 2) + '\n');
  const weights = await engine.readWeights();
  await writeFile(`${directory}/weights.f32`, new Uint8Array(weights.buffer));
  console.log(JSON.stringify(metadata));
} finally { engine.destroy(); }
