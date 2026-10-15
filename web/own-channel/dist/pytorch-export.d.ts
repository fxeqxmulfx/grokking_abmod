import type { TrainingOptions } from './types.ts';
export declare function pytorchSource(options: Pick<TrainingOptions, 'model' | 'seed' | 'maxEpochs'> & Partial<Pick<TrainingOptions, 'preset' | 'optimizer'>>): string;
