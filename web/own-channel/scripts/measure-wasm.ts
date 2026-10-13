import {root} from '../project.ts';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {WASMTrainer} from '../dist/wasm.js';
import {Sampler, ShortGrokking} from '../dist/model.js';
import {modelInfo, modelId} from '../dist/models.js';

const model = modelId(process.argv[2] || 'dense-rotary'), variant = process.argv[3] || 'simd', directory = process.argv[4];
if (!directory) throw new Error('Usage: node scripts/measure-wasm.ts <model> <simd|scalar> <output-directory>');
const bytes = await readFile(new URL(`dist/${model}-${variant}.wasm`, root));
const info = modelInfo(model), seed = Number(process.argv[5] ?? info.seed);
const engine = await WASMTrainer.create(seed, {model, bytes: new Uint8Array(bytes), scalar: variant === 'scalar'}), sampler = new Sampler(seed), criterion = new ShortGrokking(), history = [];
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
  const metadata = {model, variant, seed, embeddingInitialization:'normal-width', width:info.width, parameters:info.parameters, reportEvery:info.reportEvery, elapsedSeconds: (performance.now() - start) / 1000, wasmSha256: createHash('sha256').update(bytes).digest('hex'), node: process.version, criterion, sourceSha256:{} as Record<string, string>};
  for (const name of ['wasm.js','models.js','model.js']) metadata.sourceSha256[name] = createHash('sha256').update(await readFile(new URL(`dist/${name}`, root))).digest('hex');
  await writeFile(`${directory}/run.json`, JSON.stringify(metadata, null, 2) + '\n');
  const weights = await engine.readWeights();
  await writeFile(`${directory}/weights.f32`, new Uint8Array(weights.buffer));
  console.log(JSON.stringify(metadata));
} finally { engine.destroy(); }
