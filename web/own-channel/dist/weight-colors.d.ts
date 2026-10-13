import type { Weights } from './types.ts';
export declare function weightGradient(positive: boolean): string;
export declare function weightMagnitude(value: number, range: number): number;
export declare function weightColor(positive: boolean, magnitude: number): string;
export declare function weightPalette(levels: number): Uint8ClampedArray;
export declare function initialWeightRanges(initial: Weights, channels: number): [number, number];
