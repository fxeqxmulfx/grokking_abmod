import { errorText } from './types.js';
import { CPUScene, topologyLayout, NETWORK_LEVELS } from './topology-scene.js';
import { initialWeightRanges } from './weight-colors.js';
import { networkScale } from './render-policy.js';
export { topologyLayout };
export { weightColor, weightPalette, weightMagnitude, weightGradient, initialWeightRanges } from './weight-colors.js';
export class WeightTopology {
    element;
    invalidate;
    uploads = 0;
    worker = null;
    scene = null;
    ctx = null;
    job = 0;
    weights = null;
    initial = null;
    generation = 0;
    transferred = false;
    initializing = false;
    contextEvents = null;
    constructor(element, invalidate) {
        this.element = element;
        this.invalidate = invalidate;
        element.width = element.height = 1;
        element.dataset.thread = 'inactive';
        element.dataset.memoryBytes = element.dataset.rasterPixels = '0';
    }
    start() {
        if (this.worker || this.ctx || this.initializing)
            return;
        const generation = this.generation;
        if (typeof Worker === 'function' && typeof this.element.transferControlToOffscreen === 'function') {
            try {
                this.worker = new Worker(new URL('./topology-worker.js', import.meta.url), { type: 'module' });
                const canvas = this.element.transferControlToOffscreen();
                this.transferred = true;
                this.worker.onmessage = ({ data }) => {
                    if (generation !== this.generation)
                        return;
                    if (data.type === 'error') {
                        this.fallback(data.message);
                        return;
                    }
                    if (data.type === 'invalidate') {
                        this.invalidate();
                        return;
                    }
                    const dataset = this.element.dataset;
                    dataset.frame = String(data.job);
                    dataset.classifications = String(data.classifications);
                    dataset.backgroundPaints = String(data.backgroundPaints);
                    dataset.buildMs = String(data.buildMs);
                    dataset.drawMs = String(data.drawMs);
                    dataset.memoryBytes = String(data.memoryBytes);
                    dataset.rasterPixels = String(data.rasterPixels);
                };
                this.worker.onerror = event => { if (generation === this.generation)
                    this.fallback(event.message); };
                this.worker.postMessage({ type: 'initialize', canvas }, [canvas]);
                this.element.dataset.thread = 'worker';
            }
            catch (error) {
                this.fallback(errorText(error));
            }
        }
        else
            this.fallback();
    }
    fallback(message) {
        this.generation++;
        this.worker?.terminate();
        this.worker = null;
        // A transferred canvas cannot regain a 2D context; replace it on failure.
        if (this.transferred) {
            const replacement = this.element.cloneNode(false);
            replacement.width = replacement.height = 1;
            this.element.replaceWith(replacement);
            this.element = replacement;
        }
        this.contextEvents?.abort();
        this.contextEvents = new AbortController();
        this.element.addEventListener('contextrestored', () => this.invalidate(), { signal: this.contextEvents.signal });
        this.ctx = this.element.getContext('2d', { willReadFrequently: true });
        if (!this.ctx)
            throw new Error('Canvas 2D is unavailable.');
        this.transferred = false;
        this.scene = null;
        this.weights = this.initial = null;
        this.element.dataset.thread = 'main';
        if (message)
            this.element.dataset.error = message;
        const generation = this.generation;
        this.initializing = true;
        void CPUScene.create().then(scene => {
            if (generation !== this.generation)
                return;
            scene.onRestore = () => { if (generation === this.generation)
                this.invalidate(); };
            this.scene = scene;
            this.initializing = false;
            this.invalidate();
        })
            .catch(error => { if (generation === this.generation) {
            this.initializing = false;
            this.element.dataset.error = errorText(error);
        } });
    }
    clear() {
        const job = ++this.job;
        this.element.dataset.request = String(job);
        if (this.worker)
            this.worker.postMessage({ type: 'clear', job });
        else if (this.ctx) {
            this.ctx.setTransform(1, 0, 0, 1, 0, 0);
            this.ctx.clearRect(0, 0, this.element.width, this.element.height);
            this.element.dataset.frame = String(job);
        }
    }
    draw(_overlay, layout, weights, selected, view, focus = -1, options = {}) {
        this.start();
        // Bound the final raster dimensions; CPUScene reuses a small strip of
        // accumulation memory for the complete image.
        const initial = options.initial ?? weights, mode = options.mode ?? 'weight', scale = networkScale(layout.width, layout.height);
        const dataset = this.element.dataset, ranges = initialWeightRanges(initial, layout.channels).map(value => value * (mode === 'delta' ? .25 : 1));
        dataset.connections = String(3 * 97 * layout.channels + layout.channels * layout.channels);
        dataset.parameters = String(weights.length);
        dataset.mode = mode;
        dataset.highlight = String(Boolean(options.highlight));
        dataset.embeddingScale = String(ranges[0]);
        dataset.meshScale = String(ranges[1]);
        dataset.colorScale = 'logarithmic';
        dataset.colorLevels = String(NETWORK_LEVELS);
        dataset.renderer = 'CPU · Canvas 2D';
        dataset.precision = 'f32 accumulation';
        const job = ++this.job;
        dataset.request = String(job);
        const changed = this.weights !== weights, initialChanged = this.initial !== initial;
        if (changed)
            this.uploads++;
        if (this.worker) {
            const request = { type: 'draw', job, width: layout.width, height: layout.height, channels: layout.channels, scale, selected, view, focus, mode, highlight: Boolean(options.highlight) };
            const transfer = [];
            if (changed) {
                request.weights = weights.slice();
                transfer.push(request.weights.buffer);
            }
            if (initialChanged) {
                request.initial = initial.slice();
                transfer.push(request.initial.buffer);
            }
            this.worker.postMessage(request, transfer);
        }
        else {
            if (!this.scene)
                return;
            const width = Math.round(layout.width * scale), height = Math.round(layout.height * scale);
            if (this.element.width !== width)
                this.element.width = width;
            if (this.element.height !== height)
                this.element.height = height;
            this.scene.draw(this.ctx, layout, weights, selected, view, focus, options, scale);
            dataset.frame = String(job);
            dataset.classifications = String(this.scene.classifications);
            dataset.backgroundPaints = String(this.scene.backgroundPaints);
            dataset.buildMs = String(this.scene.buildMs);
            dataset.drawMs = String(this.scene.drawMs);
            dataset.memoryBytes = String(this.scene.exports.memory.buffer.byteLength);
            dataset.rasterPixels = String(width * height);
        }
        this.weights = weights;
        this.initial = initial;
        dataset.uploads = String(this.uploads);
    }
    release() {
        if (!this.worker && !this.ctx && !this.initializing)
            return false;
        this.generation++;
        this.worker?.terminate();
        this.worker = null;
        this.contextEvents?.abort();
        this.contextEvents = null;
        if (this.transferred) {
            // A transferred canvas cannot be transferred again after its worker exits.
            const replacement = this.element.cloneNode(false);
            replacement.width = replacement.height = 1;
            this.element.replaceWith(replacement);
            this.element = replacement;
        }
        else
            this.element.width = this.element.height = 1;
        this.transferred = false;
        this.ctx = null;
        this.scene = null;
        this.initializing = false;
        this.weights = this.initial = null;
        const dataset = this.element.dataset;
        dataset.thread = 'inactive';
        dataset.memoryBytes = dataset.rasterPixels = '0';
        delete dataset.frame;
        delete dataset.request;
        return true;
    }
    destroy() { this.release(); }
}
//# sourceMappingURL=topology.js.map