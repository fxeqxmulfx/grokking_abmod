import type { Preset } from './types.ts';
export interface OptimizerProfile {
    learningRate: number;
    weightDecay: number;
    nativeSeeds: number[];
    nativeTest99Epochs: (number | null)[];
    nativeBudget: number;
    nativePlateauEpochs?: number[];
}
export type WidthProfiles = {
    fast: OptimizerProfile;
} & Partial<Record<Preset, OptimizerProfile>>;
export declare const OPTIMIZER_PROFILES: Record<number, WidthProfiles>;
