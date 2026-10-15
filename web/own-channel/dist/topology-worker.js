import { errorText } from './types.js';
import { CPUScene, topologyLayout } from './topology-scene.js';
let canvas, ctx;
let weights, initial, pending = null;
let timer = null, awaitingPaint = null;
let legacyPresentation = false;
const ready = CPUScene.create();
void ready.then(scene => { scene.onRestore = () => postMessage({ type: 'invalidate' }); }).catch(() => { });
function schedule() {
    if (timer !== null || awaitingPaint !== null || !pending)
        return;
    timer = setTimeout(async () => {
        try {
            const scene = await ready, request = pending;
            pending = null;
            if (request.type === 'clear') {
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            else {
                const width = Math.round(request.width * request.scale), height = Math.round(request.height * request.scale);
                if (canvas.width !== width)
                    canvas.width = width;
                if (canvas.height !== height)
                    canvas.height = height;
                scene.draw(ctx, topologyLayout(request.width, request.height, request.channels), weights, request.selected, request.view, request.focus, { initial, mode: request.mode, highlight: request.highlight }, request.scale);
            }
            if (ctx.isContextLost())
                throw new Error('The network drawing context was lost.');
            const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
            if (!legacyPresentation)
                awaitingPaint = request.job;
            postMessage({ type: 'paint', job: request.job, width: image.width, height: image.height, pixels: image.data,
                classifications: scene.classifications, backgroundPaints: scene.backgroundPaints, buildMs: scene.buildMs, drawMs: scene.drawMs,
                memoryBytes: scene.exports.memory.buffer.byteLength, rasterPixels: canvas.width * canvas.height }, [image.data.buffer]);
        }
        catch (error) {
            postMessage({ type: 'error', message: errorText(error) });
        }
        finally {
            timer = null;
        }
    }, 0);
}
onmessage = ({ data }) => {
    try {
        if (data.type === 'initialize') {
            // Pages open before deployment can still send the previous protocol.
            // Current clients keep their visible canvas on the main thread.
            legacyPresentation = Boolean(data.canvas);
            canvas = data.canvas ?? new OffscreenCanvas(1, 1);
            const context = canvas.getContext('2d', { willReadFrequently: true });
            if (!context)
                throw new Error('Canvas 2D is unavailable.');
            ctx = context;
            canvas.addEventListener('contextrestored', () => postMessage({ type: 'invalidate' }));
            return;
        }
        if (data.type === 'presented') {
            if (data.job === awaitingPaint) {
                awaitingPaint = null;
                schedule();
            }
            return;
        }
        // Retain snapshots even when a newer pose replaces a queued draw.
        if (data.type === 'draw') {
            if (data.weights)
                weights = data.weights;
            if (data.initial)
                initial = data.initial;
        }
        pending = data;
        schedule();
    }
    catch (error) {
        postMessage({ type: 'error', message: errorText(error) });
    }
};
//# sourceMappingURL=topology-worker.js.map