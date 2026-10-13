import {errorText} from './types.ts';
import type {TrainingEngine, TrainingOptions, WorkerRequest, WorkerResponse} from './types.ts';
declare function postMessage(message: WorkerResponse, transfer: Transferable[]): void;
const post = (message: WorkerResponse, transfer: Transferable[] = []) => postMessage(message, transfer);
import {Sampler, ShortGrokking} from './model.ts';
import {modelInfo} from './models.ts';
import {createEngine} from './engine.ts';

let engine: TrainingEngine | undefined, sampler: Sampler, criterion: ShortGrokking;
let paused = true, running = false;
let options: TrainingOptions, activeMs = 0, lastSnapshot = 0, lastSnapshotStep = -1, chunk = 2;

async function snapshot() {
  if (engine!.step === lastSnapshotStep) return;
  const weights = await engine!.readWeights();
  if (!weights.every(Number.isFinite)) throw new Error('Training diverged: weights are not finite.');
  post({type: 'weights', step: engine!.step, weights, activeMs}, [weights.buffer]);
  lastSnapshot = performance.now(); lastSnapshotStep = engine!.step;
}

async function report() {
  const row = await engine!.evaluate(); row.phase = criterion.observe(row);
  post({type: 'metrics', row, criterion});
  return row;
}

async function run(singleReport = false) {
  if (running || !engine) return;
  running = true; paused = false; post({type: 'status', status: 'Training'});
  const cap = options.maxEpochs * 10, interval = modelInfo(options.model).reportEvery;
  const limit = singleReport ? Math.min(cap, (Math.floor(engine.step / interval) + 1) * interval) : cap;
  let status = 'Paused';
  try {
    while (!paused && engine.step < limit) {
      const boundary = Math.min(limit, (Math.floor(engine.step / interval) + 1) * interval);
      const count = Math.min(chunk, boundary - engine.step);
      const samples = Array.from({length: count}, () => sampler.sample().slice());
      const start = performance.now();
      await engine.train(samples);
      const elapsed = performance.now() - start;
      activeMs += elapsed;
      // Keep each submission near 50 ms so pause and capture stay responsive.
      chunk = Math.max(1, Math.min(32, Math.round(count * 50 / Math.max(elapsed, 1))));
      if (engine.step === boundary) {
        const row = await report();
        if (options.stopAtTarget && row.heldoutAccuracy >= .99) { status = criterion.event?.delayed ? 'Grokking reached' : 'Target reached'; break; }
      }
      if (performance.now() - lastSnapshot >= 1000 / options.captureHz) await snapshot();
    }
    if (engine.step >= cap) status = 'Epoch limit reached';
    await snapshot(); post({type: 'status', status});
  } finally { running = false; paused = true; }
}

onmessage = async ({data: message}: MessageEvent<WorkerRequest>) => {
  try {
    if (message.type === 'initialize') {
      options = message.options;
      activeMs = lastSnapshot = 0; lastSnapshotStep = -1; chunk = 2;
      if (!Number.isInteger(options.seed) || options.seed < 0 || options.seed > 0xffffffff) throw new Error('Seed must be an integer between 0 and 4294967295.');
      if (!Number.isInteger(options.maxEpochs) || options.maxEpochs < 1 || options.maxEpochs > 10000) throw new Error('Epoch limit must be between 1 and 10000.');
      if (![1, 5, 10].includes(options.captureHz)) throw new Error('Snapshot rate must be 1, 5 or 10 Hz.');
      engine = await createEngine(options.seed, options.backend, options.model);
      sampler = new Sampler(options.seed); criterion = new ShortGrokking();
      await report(); await snapshot(); post({type: 'ready', info: engine.info()});
      post({type: 'status', status: 'Ready'});
    } else if (message.type === 'train') {
      await run();
    } else if (message.type === 'step') {
      await run(true);
    } else if (message.type === 'pause') {
      paused = true;
    } else if (message.type === 'configure' && !running) {
      if (!Number.isInteger(message.maxEpochs) || message.maxEpochs < 1 || message.maxEpochs > 10000) throw new Error('Epoch limit must be between 1 and 10000.');
      options.maxEpochs = message.maxEpochs;
      options.stopAtTarget = Boolean(message.stopAtTarget);
    } else if (message.type === 'capture') {
      if (![1, 5, 10].includes(message.hz)) throw new Error('Snapshot rate must be 1, 5 or 10 Hz.');
      options.captureHz = message.hz;
    }
  } catch (error) {
    paused = true;
    post({type: 'error', message: errorText(error)});
  }
};
