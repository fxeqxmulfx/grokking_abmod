import type {Preset} from './types.ts';

// Confirmed Rust/CUDA profiles. Fast is the optimizer grid's winner, the
// earliest 99% held-out crossing on every confirmation seed. A
// late-generalization search ranks the configurations that succeed on every
// seed: slow crosses latest, and plateau has the longest memorization plateau
// (the website's ShortGrokking rule) on its shortest seed. Where that plateau
// belongs to the slow configuration, plateau takes the longest one that groks
// by the same rule. Plateau lengths are listed wherever every seed holds one.
export interface OptimizerProfile {
  learningRate: number; weightDecay: number; nativeSeeds: number[]; nativeTest99Epochs: (number | null)[]; nativeBudget: number;
  nativePlateauEpochs?: number[];
}
export type WidthProfiles = {fast: OptimizerProfile} & Partial<Record<Preset, OptimizerProfile>>;
const profile = (learningRate: number, weightDecay: number, nativeTest99Epochs: (number | null)[], nativePlateauEpochs?: number[]): OptimizerProfile =>
  ({learningRate, weightDecay, nativeSeeds: [11, 17, 23], nativeTest99Epochs, nativeBudget: 10000, ...nativePlateauEpochs && {nativePlateauEpochs}});
export const OPTIMIZER_PROFILES: Record<number, WidthProfiles> = {
  16: {fast: profile(0.1, 0, [1033, 407, 163]), slow: profile(0.2, 0.001, [5182, 1333, 7117])},
  32: {fast: profile(0.1, 0.0001, [39, 43, 27]), slow: profile(0.0003, 0.1, [5199, 3885, 5462])},
  64: {fast: profile(0.1, 0.1, [21, 33, 29]), slow: profile(0.0001, 0.0001, [6513, 6583, 8501])},
  128: {fast: profile(0.01, 1, [30, 32, 30]), plateau: profile(0.0003, 0, [3375, 3770, 2880], [166, 150, 210]), slow: profile(0.00005, 0.3, [8151, 7931, 8899])},
  256: {fast: profile(0.01, 1, [32, 33, 27]), plateau: profile(0.001, 0.3, [368, 346, 336], [76, 69, 58]), slow: profile(0.00005, 0.3, [7632, 6777, 6593], [1644, 1403, 1362])},
};
