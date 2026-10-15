// Trains one website model with the browser's WASM engine in Node, stopping at
// 99% held-out accuracy like the site's auto-pause, and saves the history.
// Usage: node wasm-run.mjs <model> <lr> <wd> <seed> <threads> <max-epochs> <out.json>
import {readFile, writeFile} from 'node:fs/promises';
const B = '/home/misha/rust_projects/grokking_abmod/web/own-channel/build/';
const {WASMTrainer} = await import(B + 'dist/wasm.js');
const {ParallelWASMTrainer} = await import(B + 'dist/parallel-wasm.js');
const {Sampler, ShortGrokking} = await import(B + 'dist/model.js');
const {modelInfo, modelId} = await import(B + 'dist/models.js');
const {nodeCPUThread} = await import(B + 'tests/node-cpu.js');
const [name, lr, wd, seedText, threadsText, maxText, out] = process.argv.slice(2);
const model = modelId(name), info = modelInfo(model), seed = Number(seedText), threads = Number(threadsText), maxEpochs = Number(maxText);
const options = {model, scalar: false, learningRate: Number(lr), decay: Number(wd)};
const bytes = new Uint8Array(await readFile(`${B}dist/${model}-simd${threads > 1 ? '-shared' : ''}.wasm`));
const engine = threads > 1 ? await ParallelWASMTrainer.create(seed, {...options, threads, bytes, workerFactory: nodeCPUThread}) : await WASMTrainer.create(seed, {...options, bytes});
const sampler = new Sampler(seed), criterion = new ShortGrokking(), rows = [];
const start = performance.now();
let first = null;
try {
  for (;;) {
    const row = await engine.evaluate(); row.phase = criterion.observe(row); rows.push([row.epoch, row.trainAccuracy, row.heldoutAccuracy, row.trainLoss, row.heldoutLoss]);
    if (row.epoch % 500 === 0) console.log(`${model} lr ${lr} wd ${wd} seed ${seed} t${threads}: epoch ${row.epoch} train ${(100 * row.trainAccuracy).toFixed(2)}% test ${(100 * row.heldoutAccuracy).toFixed(2)}% (${((performance.now() - start) / 1000).toFixed(0)} s)`);
    if (row.heldoutAccuracy >= .99) { first = row.epoch; break; }
    if (row.epoch >= maxEpochs) break;
    for (let n = 0; n < info.reportEvery; n += 10) await engine.train(Array.from({length: 10}, () => sampler.sample().slice()));
  }
} finally { engine.destroy(); }
const seconds = (performance.now() - start) / 1000;
const train99 = rows.find(r => r[1] >= .99)?.[0] ?? null;
await writeFile(out, JSON.stringify({model, width: info.width, learningRate: Number(lr), weightDecay: Number(wd), seed, threads, maxEpochs, firstTrain99: train99, firstTest99: first, plateauEpochs: criterion.plateau.epochs, event: criterion.event, seconds, node: process.version, rows}) + '\n');
console.log(`DONE ${model} lr ${lr} wd ${wd} seed ${seed} t${threads}: train99 ${train99}, test99 ${first}, plateau ${criterion.plateau.epochs}, ${seconds.toFixed(0)} s`);
