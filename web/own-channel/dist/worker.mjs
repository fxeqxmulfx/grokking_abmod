import {Sampler, Grokking, ShortGrokking} from './model.mjs';
import {modelInfo} from './models.mjs';
import {createEngine} from './engine.mjs';

let engine, sampler, criterion, paused = true, running = false;
let options = {}, activeMs = 0, lastSnapshot = 0, chunk = 2;

async function snapshot() {
  const weights = await engine.readWeights();
  if (!weights.every(Number.isFinite)) throw new Error('Training diverged: weights are not finite.');
  postMessage({type: 'weights', step: engine.step, weights, activeMs}, [weights.buffer]);
  lastSnapshot = performance.now();
}

async function report() {
  const row = await engine.evaluate(); row.phase = criterion.observe(row);
  postMessage({type: 'metrics', row, criterion});
  return row;
}

async function run(singleReport = false) {
  if (running || !engine) return;
  running = true; paused = false; postMessage({type: 'status', status: 'Training'});
  const cap = options.maxEpochs * 10, interval = modelInfo(options.model ?? 'concat').reportEvery;
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
        if (options.stopAtTarget && row.heldoutAccuracy >= .99) { status = criterion.event.delayed ? 'Grokking reached' : 'Target reached'; break; }
      }
      if (performance.now() - lastSnapshot >= 1000 / options.captureHz) await snapshot();
    }
    if (engine.step >= cap) status = 'Epoch limit reached';
    await snapshot(); postMessage({type: 'status', status});
  } finally { running = false; paused = true; }
}

onmessage = async ({data: message}) => {
  try {
    if (message.type === 'initialize') {
      options = {backend: 'auto', maxEpochs: 10000, captureHz: 5, stopAtTarget: true, ...message.options};
      if (!Number.isInteger(options.seed) || options.seed < 0 || options.seed > 0xffffffff) throw new Error('Seed must be an integer between 0 and 4294967295.');
      if (!Number.isInteger(options.maxEpochs) || options.maxEpochs < 1 || options.maxEpochs > 10000) throw new Error('Epoch limit must be between 1 and 10000.');
      if (![1, 5, 10].includes(options.captureHz)) throw new Error('Snapshot rate must be 1, 5 or 10 Hz.');
      engine = await createEngine(options.seed, options.backend, options.model ?? 'concat');
      sampler = new Sampler(options.seed); criterion = modelInfo(options.model ?? 'concat').plateauEpochs ? new ShortGrokking() : new Grokking();
      await report(); await snapshot(); postMessage({type: 'ready', info: engine.info()});
      postMessage({type: 'status', status: 'Ready'});
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
      options.captureHz = message.hz;
    }
  } catch (error) {
    paused = true;
    postMessage({type: 'error', message: error.message || String(error)});
  }
};
