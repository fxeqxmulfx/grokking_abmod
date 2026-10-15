import { errorText } from './types.js';
import { CPUScene, topologyLayout, NETWORK_LEVELS } from './topology-scene.js';
import { initialWeightRanges } from './weight-colors.js';
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
    constructor(element, invalidate) {
        this.element = element;
        this.invalidate = invalidate;
        if (typeof Worker === 'function' && typeof element.transferControlToOffscreen === 'function') {
            try {
                this.worker = new Worker(new URL('./topology-worker.js', import.meta.url), { type: 'module' });
                const canvas = element.transferControlToOffscreen();
                this.worker.onmessage = ({ data }) => {
                    if (data.type === 'error') {
                        this.fallback(data.message);
                        return;
                    }
                    const dataset = this.element.dataset;
                    dataset.frame = String(data.job);
                    dataset.classifications = String(data.classifications);
                    dataset.backgroundPaints = String(data.backgroundPaints);
                    dataset.buildMs = String(data.buildMs);
                    dataset.drawMs = String(data.drawMs);
                };
                this.worker.onerror = event => this.fallback(event.message);
                this.worker.postMessage({ type: 'initialize', canvas }, [canvas]);
                element.dataset.thread = 'worker';
            }
            catch (error) {
                this.fallback(errorText(error));
            }
        }
        else
            this.fallback();
    }
    fallback(message) {
        this.worker?.terminate();
        this.worker = null;
        // A transferred canvas cannot regain a 2D context; replace it on failure.
        try {
            this.ctx = this.element.getContext('2d', { willReadFrequently: true });
        }
        catch {
            const replacement = this.element.cloneNode(false);
            this.element.replaceWith(replacement);
            this.element = replacement;
            this.ctx = replacement.getContext('2d', { willReadFrequently: true });
        }
        if (!this.ctx)
            throw new Error('Canvas 2D is unavailable.');
        this.scene = null;
        this.weights = this.initial = null;
        this.element.dataset.thread = 'main';
        if (message)
            this.element.dataset.error = message;
        // Constructors run before the app's render functions are initialized.
        void CPUScene.create().then(scene => { this.scene = scene; this.invalidate(); }).catch(error => { this.element.dataset.error = errorText(error); });
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
        // Dense connections need less resolution than the separate text overlay.
        // Bound the f32 accumulation buffer when a tab expands to the full screen.
        const initial = options.initial ?? weights, mode = options.mode ?? 'weight', scale = Math.min(devicePixelRatio || 1, 2, Math.sqrt(524288 / (layout.width * layout.height)));
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
        }
        this.weights = weights;
        this.initial = initial;
        dataset.uploads = String(this.uploads);
    }
    destroy() { this.worker?.terminate(); this.worker = null; this.scene = null; }
}
//# sourceMappingURL=topology.js.map