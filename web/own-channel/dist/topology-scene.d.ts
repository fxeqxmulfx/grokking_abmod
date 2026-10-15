import type { Weights, NetworkView, WeightMode } from './types.ts';
export interface TopologyLayout {
    width: number;
    height: number;
    channels: number;
    rows: [number, number, number, number];
    column(c: number): number;
    residue(r: number, role: number): number;
    answer(r: number): number;
}
export interface DrawOptions {
    initial?: Weights;
    mode?: WeightMode;
    highlight?: boolean;
}
type Context = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
export interface RasterExports {
    memory: WebAssembly.Memory;
    render(codes: number, palette: number, columns: number, residues: number, answers: number, scratch: number, pixels: number, width: number, height: number, channels: number, y0: number, y1: number, y2: number, y3: number): void;
}
export declare const NETWORK_LEVELS = 1024;
export declare function topologyLayout(width: number, height: number, channels: number): TopologyLayout;
export declare class CPUScene {
    readonly exports: RasterExports;
    static create(scalar?: boolean, bytes?: Uint8Array<ArrayBuffer>): Promise<CPUScene>;
    classifications: number;
    backgroundPaints: number;
    buildMs: number;
    drawMs: number;
    onRestore?: () => void;
    private weights;
    private initial;
    private mode;
    private addresses;
    private codes;
    private palette;
    private coordinates;
    private image;
    private background;
    private backgroundContext;
    private shape;
    private pose;
    constructor(exports: RasterExports);
    private allocate;
    private prepare;
    draw(ctx: Context, layout: TopologyLayout, weights: Weights, selected: number, view: NetworkView, focus?: number, options?: DrawOptions, scale?: number): void;
    private highlight;
}
export {};
