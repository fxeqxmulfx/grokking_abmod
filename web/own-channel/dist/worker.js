import { errorText } from './types.js';
const post = (message, transfer = []) => postMessage(message, transfer);
import { Sampler, ShortGrokking } from './model.js';
import { modelInfo } from './models.js';
import { createEngine } from './engine.js';
import { AccuracyPause } from './training-control.js';
let engine, sampler, criterion;
let paused = true, running = false;
let target = new AccuracyPause();
let options, activeMs = 0, lastSnapshot = 0, lastSnapshotStep = -1, chunk = 2;
let task = Promise.resolve(), suspending = false, currentStatus = 'Loading';
const status = (value) => { currentStatus = value; post({ type: 'status', status: value }); };
async function snapshot() {
    if (engine.step === lastSnapshotStep)
        return;
    const weights = await engine.readWeights();
    if (!weights.every(Number.isFinite))
        throw new Error('Training diverged: weights are not finite.');
    post({ type: 'weights', step: engine.step, weights, activeMs }, [weights.buffer]);
    lastSnapshot = performance.now();
    lastSnapshotStep = engine.step;
}
async function report() {
    const row = await engine.evaluate();
    row.phase = criterion.observe(row);
    post({ type: 'metrics', row, criterion });
    return row;
}
async function run(singleReport = false) {
    if (running || !engine)
        return;
    running = true;
    paused = false;
    status('Training');
    const cap = options.maxEpochs * 10, interval = modelInfo(options.model).reportEvery;
    const limit = singleReport ? Math.min(cap, (Math.floor(engine.step / interval) + 1) * interval) : cap;
    let finalStatus = 'Paused';
    try {
        while (!paused && engine.step < limit) {
            const boundary = Math.min(limit, (Math.floor(engine.step / interval) + 1) * interval);
            const count = Math.min(chunk, boundary - engine.step);
            const samples = Array.from({ length: count }, () => sampler.sample().slice());
            const start = performance.now();
            await engine.train(samples);
            const elapsed = performance.now() - start;
            activeMs += elapsed;
            // Keep each submission near 50 ms so pause and capture stay responsive.
            chunk = Math.max(1, Math.min(32, Math.round(count * 50 / Math.max(elapsed, 1))));
            if (engine.step === boundary) {
                const row = await report();
                if (target.observe(row.heldoutAccuracy)) {
                    finalStatus = 'Paused at 99% held-out accuracy';
                    break;
                }
            }
            if (performance.now() - lastSnapshot >= 1000 / options.captureHz)
                await snapshot();
        }
        if (engine.step >= cap && finalStatus === 'Paused')
            finalStatus = 'Epoch limit reached';
        await snapshot();
        status(finalStatus);
    }
    finally {
        running = false;
        paused = true;
    }
}
async function initialize(settings, checkpoint) {
    options = settings;
    activeMs = lastSnapshot = 0;
    lastSnapshotStep = -1;
    chunk = 2;
    target = new AccuracyPause();
    if (!Number.isInteger(options.seed) || options.seed < 0 || options.seed > 0xffffffff)
        throw new Error('Seed must be an integer between 0 and 4294967295.');
    if (!Number.isInteger(options.maxEpochs) || options.maxEpochs < 1 || options.maxEpochs > 10000)
        throw new Error('Epoch limit must be between 1 and 10000.');
    if (![1, 5, 10].includes(options.captureHz))
        throw new Error('Snapshot rate must be 1, 5 or 10 Hz.');
    engine = await createEngine(options.seed, options.backend, options.model, options.preset, options.optimizer);
    sampler = new Sampler(options.seed);
    criterion = new ShortGrokking();
    if (checkpoint) {
        engine.restore(checkpoint.engine);
        sampler.order = checkpoint.sampler.order;
        sampler.position = checkpoint.sampler.position;
        sampler.random.low = checkpoint.sampler.low;
        sampler.random.high = checkpoint.sampler.high;
        Object.assign(criterion, checkpoint.criterion);
        target.reached = checkpoint.targetReached;
        activeMs = checkpoint.activeMs;
        lastSnapshotStep = engine.step;
    }
    else {
        await report();
        await snapshot();
    }
    post({ type: 'ready', info: engine.info() });
    status(checkpoint?.status ?? 'Ready');
}
async function suspend() {
    if (engine) {
        const checkpoint = {
            engine: engine.checkpoint(),
            sampler: { order: sampler.order, position: sampler.position, low: sampler.random.low, high: sampler.random.high },
            criterion, targetReached: target.reached, activeMs, status: currentStatus,
        };
        engine.destroy();
        engine = undefined;
        post({ type: 'suspended', checkpoint }, [checkpoint.engine.weights.buffer, checkpoint.engine.moment1.buffer, checkpoint.engine.moment2.buffer, checkpoint.sampler.order.buffer]);
    }
    else
        post({ type: 'suspended' });
    close();
}
function perform(action) {
    task = action().catch(error => { paused = true; post({ type: 'error', message: errorText(error) }); });
}
onmessage = ({ data: message }) => {
    try {
        if (message.type === 'suspend') {
            if (suspending)
                return;
            paused = true;
            suspending = true;
            task = task.then(suspend).catch(error => { post({ type: 'error', message: errorText(error) }); post({ type: 'suspended' }); close(); });
        }
        else if (suspending)
            return;
        else if (message.type === 'initialize') {
            perform(() => initialize(message.options, message.checkpoint));
        }
        else if ((message.type === 'train' || message.type === 'step') && !running) {
            perform(() => run(message.type === 'step'));
        }
        else if (message.type === 'pause') {
            paused = true;
        }
        else if (message.type === 'configure' && !running) {
            if (!Number.isInteger(message.maxEpochs) || message.maxEpochs < 1 || message.maxEpochs > 10000)
                throw new Error('Epoch limit must be between 1 and 10000.');
            options.maxEpochs = message.maxEpochs;
        }
        else if (message.type === 'capture') {
            if (![1, 5, 10].includes(message.hz))
                throw new Error('Snapshot rate must be 1, 5 or 10 Hz.');
            options.captureHz = message.hz;
        }
    }
    catch (error) {
        paused = true;
        post({ type: 'error', message: errorText(error) });
    }
};
//# sourceMappingURL=worker.js.map