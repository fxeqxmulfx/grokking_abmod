import type { Metrics, Interval, GrokkingEvent, Sample } from './types.ts';
export declare const P = 97;
export declare const TRAIN: number, TOTAL: number;
export declare class Random {
    low: number;
    high: number;
    resultHigh: number;
    constructor(seed: bigint);
    next(): number;
    unit(): number;
    index(bound: number): number;
    shuffle<T>(values: {
        length: number;
        [index: number]: T;
    }): void;
}
export declare function dataset(seed?: number): Uint32Array<ArrayBuffer>;
export declare class Sampler {
    random: Random;
    order: Uint32Array<ArrayBuffer>;
    position: number;
    constructor(seed?: number);
    sample(batch?: number): Uint32Array<ArrayBuffer>;
}
export declare function packBatch(data: Uint32Array, sample: Sample, buffer: Uint32Array, offset?: number): {
    rows: number;
    offset: number;
    bins: number;
    end: number;
};
export declare class ShortGrokking {
    plateau: Interval;
    perfect: Interval;
    start: number | null;
    perfectStart: number | null;
    previous: number | null;
    lastLow: number | null;
    first90: number | null;
    rise: number | null;
    event: GrokkingEvent | null;
    constructor();
    observe(row: Pick<Metrics, 'epoch' | 'trainAccuracy' | 'heldoutAccuracy'>): "Generalized" | "Generalizing" | "Learning" | "Memorization plateau observed" | "Short-plateau grokking";
}
