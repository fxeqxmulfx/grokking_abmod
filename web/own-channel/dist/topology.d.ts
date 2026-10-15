import type { Weights, NetworkView } from './types.ts';
import { topologyLayout } from './topology-scene.ts';
import type { TopologyLayout, DrawOptions } from './topology-scene.ts';
export { topologyLayout };
export type { TopologyLayout };
export { weightColor, weightPalette, weightMagnitude, weightGradient, initialWeightRanges } from './weight-colors.ts';
export declare class WeightTopology {
    element: HTMLCanvasElement;
    private invalidate;
    uploads: number;
    private worker;
    private scene;
    private ctx;
    private job;
    private weights;
    private initial;
    private generation;
    private initializing;
    private contextEvents;
    constructor(element: HTMLCanvasElement, invalidate: () => void);
    private start;
    private fallback;
    clear(): void;
    draw(_overlay: CanvasRenderingContext2D, layout: TopologyLayout, weights: Weights, selected: number, view: NetworkView, focus?: number, options?: DrawOptions): void;
    release(): boolean;
    destroy(): void;
}
