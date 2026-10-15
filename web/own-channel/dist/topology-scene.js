import { weightColor, weightMagnitude, weightPalette, initialWeightRanges } from './weight-colors.js';
export const NETWORK_LEVELS = 1024;
const LEVELS = NETWORK_LEVELS, SKIP = 65535, ZERO = 4 * LEVELS, P = 97;
const TILE_PIXELS = 131072;
const rgb = weightPalette(LEVELS);
export function topologyLayout(width, height, channels) {
    const top = Math.min(26, height * .15), bottom = height - 22, span = bottom - top, half = (width - 64) / 2;
    return { width, height, channels, rows: [top, top + span * .34, top + span * .67, bottom],
        column: c => 20 + c / (channels - 1) * (width - 40),
        residue: (r, role) => 20 + role * (half + 24) + r / 96 * half,
        answer: r => 20 + r / 96 * (width - 40) };
}
// Rasterize every connection in WASM. Span accumulation is independent of
// Canvas path intersections and draw order, keeping dense zoomed views fast.
export class CPUScene {
    exports;
    static async create(scalar = false, bytes) {
        const load = async (variant) => {
            const response = await fetch(new URL(`./topology-${variant}.wasm`, import.meta.url));
            if (!response.ok)
                throw new Error(`Could not load CPU rasterizer: HTTP ${response.status}.`);
            return new Uint8Array(await response.arrayBuffer());
        };
        let source = bytes ?? await load(scalar ? 'scalar' : 'simd');
        if (!WebAssembly.validate(source) && !bytes && !scalar)
            source = await load('scalar');
        const { instance } = await WebAssembly.instantiate(source);
        return new CPUScene(instance.exports);
    }
    classifications = 0;
    backgroundPaints = 0;
    buildMs = 0;
    drawMs = 0;
    onRestore;
    weights = null;
    initial = null;
    mode = 'weight';
    addresses = { columns: 0, residues: 0, answers: 0, palette: 0, codes: 0, scratch: 0, pixels: 0 };
    codes = new Uint16Array();
    palette = new Float32Array();
    coordinates = new Float32Array();
    image = null;
    background = null;
    backgroundContext = null;
    shape = '';
    pose = '';
    constructor(exports) {
        this.exports = exports;
    }
    allocate(width, height, channels, parameters) {
        const shape = `${width}:${height}:${channels}:${parameters}`;
        if (this.shape === shape)
            return;
        const changedLayout = this.codes.length !== parameters || this.coordinates.length !== channels + 3 * P;
        const a = this.addresses;
        a.columns = 0;
        a.residues = channels * 4;
        a.answers = (channels + 2 * P) * 4;
        a.palette = Math.ceil((channels + 3 * P) * 4 / 16) * 16;
        a.codes = a.palette + (ZERO + 1) * 16;
        a.scratch = Math.ceil((a.codes + parameters * 2) / 16) * 16;
        a.pixels = a.scratch + (width + 1) * height * 16;
        const memory = this.exports.memory, needed = Math.ceil((a.pixels + width * height * 4) / 65536);
        if (needed > memory.buffer.byteLength / 65536)
            memory.grow(needed - memory.buffer.byteLength / 65536);
        this.coordinates = new Float32Array(memory.buffer, 0, channels + 3 * P);
        this.palette = new Float32Array(memory.buffer, a.palette, (ZERO + 1) * 4);
        this.codes = new Uint16Array(memory.buffer, a.codes, parameters);
        this.image = new ImageData(new Uint8ClampedArray(memory.buffer, a.pixels, width * height * 4), width, height);
        if (changedLayout)
            this.weights = null;
        this.shape = shape;
        this.pose = '';
    }
    prepare(layout, weights, initial, mode) {
        if (this.weights === weights && this.initial === initial && this.mode === mode)
            return;
        const start = performance.now(), e = P * layout.channels;
        const ranges = initialWeightRanges(initial, layout.channels).map(value => value * (mode === 'delta' ? .25 : 1));
        for (let i = 0; i < weights.length; i++) {
            const value = weights[i] - (mode === 'delta' ? initial[i] : 0);
            this.codes[i] = value === 0 ? (mode === 'delta' ? SKIP : ZERO)
                : (i >= e ? 2 * LEVELS : 0) + (value < 0 ? LEVELS : 0) + Math.min(LEVELS - 1, Math.floor(weightMagnitude(value, ranges[i >= e ? 1 : 0]) * LEVELS));
        }
        for (let group = 0; group < 2; group++)
            for (let sign = 0; sign < 2; sign++)
                for (let level = 0; level < LEVELS; level++) {
                    const index = (group * 2 * LEVELS + sign * LEVELS + level) * 4, color = (sign * LEVELS + level) * 4;
                    const magnitude = (level + .5) / LEVELS;
                    // Accumulate in f32 before converting to RGBA8; weak edges cannot be
                    // rounded away individually, even when millions of paths overlap.
                    const alpha = Math.max(1 / 4096, Math.min(.05, 1.8 / Math.sqrt(group ? layout.channels ** 2 : e)) * (.03 + .97 * magnitude));
                    for (let c = 0; c < 3; c++)
                        this.palette[index + c] = rgb[color + c] * alpha;
                    this.palette[index + 3] = alpha;
                }
        this.palette.set([157 / 4096, 161 / 4096, 150 / 4096, 1 / 4096], ZERO * 4);
        this.weights = weights;
        this.initial = initial;
        this.mode = mode;
        this.pose = '';
        this.classifications++;
        this.buildMs = performance.now() - start;
    }
    draw(ctx, layout, weights, selected, view, focus = -1, options = {}, scale = 1) {
        const start = performance.now(), width = Math.round(layout.width * scale), height = Math.round(layout.height * scale);
        // Reuse one strip of f32 scratch space instead of allocating it for the
        // complete high-resolution image. Every connection is still rasterized.
        const tileHeight = Math.min(height, Math.max(1, Math.floor(TILE_PIXELS / (width + 1))));
        this.allocate(width, tileHeight, layout.channels, weights.length);
        this.prepare(layout, weights, options.initial ?? weights, options.mode ?? 'weight');
        if (!this.background) {
            this.background = typeof OffscreenCanvas === 'function' ? new OffscreenCanvas(width, height) : document.createElement('canvas');
            this.backgroundContext = this.background.getContext('2d', { willReadFrequently: true });
            if (!this.backgroundContext)
                throw new Error('Canvas 2D is unavailable.');
            this.background.addEventListener('contextrestored', () => { this.pose = ''; this.onRestore?.(); });
        }
        const pose = `${this.classifications}:${width}:${height}:${view.zoom}:${view.x}:${view.y}`;
        if (this.pose !== pose) {
            if (this.background.width !== width)
                this.background.width = width;
            if (this.background.height !== height)
                this.background.height = height;
            const x = (value) => ((value - layout.width / 2) * view.zoom + layout.width / 2 + view.x) * scale;
            const w = layout.channels;
            for (let c = 0; c < w; c++)
                this.coordinates[c] = x(layout.column(c));
            for (let r = 0; r < P; r++) {
                this.coordinates[w + r] = x(layout.residue(r, 0));
                this.coordinates[w + P + r] = x(layout.residue(r, 1));
                this.coordinates[w + 2 * P + r] = x(layout.answer(r));
            }
            const a = this.addresses, rows = layout.rows.map(y => (y + view.y) * scale);
            for (let top = 0; top < height; top += tileHeight) {
                this.exports.render(a.codes, a.palette, a.columns, a.residues, a.answers, a.scratch, a.pixels, width, tileHeight, w, rows[0] - top, rows[1] - top, rows[2] - top, rows[3] - top);
                this.backgroundContext.putImageData(this.image, 0, top);
            }
            this.pose = pose;
            this.backgroundPaints++;
        }
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(this.background, 0, 0);
        if (options.highlight && (selected >= 0 || focus >= 0))
            this.highlight(ctx, layout, selected, focus, view, scale);
        this.drawMs = performance.now() - start;
    }
    highlight(ctx, layout, selected, focus, view, scale) {
        const { channels: w, rows } = layout, e = P * w, paths = new Map();
        const line = (index, negative, ax, ay, bx, by) => {
            let code = this.codes[index];
            if (code === SKIP)
                return;
            if (negative && code < ZERO)
                code ^= LEVELS;
            let path = paths.get(code);
            if (!path) {
                path = new Path2D();
                paths.set(code, path);
            }
            path.moveTo(ax, ay);
            path.lineTo(bx, by);
        };
        const embedding = (r, c, role) => {
            if (role === 0)
                line(r * w + c, false, layout.residue(r, 0), rows[0], layout.column(c), rows[1]);
            else if (role === 1)
                line(r * w + (c ^ 1), !(c % 2), layout.residue(r, 1), rows[0], layout.column(c), rows[1]);
            else
                line(r * w + c, false, layout.column(c), rows[2], layout.answer(r), rows[3]);
        };
        if (focus >= 0 && focus < e) {
            const r = Math.floor(focus / w), c = focus % w;
            embedding(r, c, 0);
            embedding(r, c ^ 1, 1);
            embedding(r, c, 2);
        }
        else if (focus >= e && focus < this.codes.length)
            line(focus, false, layout.column(Math.floor((focus - e) / w)), rows[1], layout.column((focus - e) % w), rows[2]);
        else if (selected >= 0 && selected < w) {
            for (let r = 0; r < P; r++)
                for (let role = 0; role < 3; role++)
                    embedding(r, selected, role);
            for (let i = 0; i < w; i++)
                line(e + i * w + selected, false, layout.column(i), rows[1], layout.column(selected), rows[2]);
        }
        ctx.save();
        ctx.setTransform(scale, 0, 0, scale, 0, 0);
        ctx.translate(layout.width / 2 + view.x, view.y);
        ctx.scale(view.zoom, 1);
        ctx.translate(-layout.width / 2, 0);
        for (const [code, path] of paths) {
            const mesh = code >= 2 * LEVELS && code < ZERO, magnitude = ((code % LEVELS) + .5) / LEVELS;
            ctx.strokeStyle = code === ZERO ? '#9da196' : weightColor(code % (2 * LEVELS) < LEVELS, magnitude);
            ctx.lineWidth = mesh ? Math.max(1 / scale, Math.min(.5 + magnitude * 2.2, (layout.width - 40) / w * .4)) : .25 + magnitude * .85;
            ctx.globalAlpha = mesh ? 1 : .26;
            ctx.stroke(path);
        }
        ctx.restore();
    }
}
//# sourceMappingURL=topology-scene.js.map