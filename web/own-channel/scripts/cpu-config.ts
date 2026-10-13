// Measured kernel choices. The optimizer changes these only after exact
// single-thread numerical validation and a separate confirmation benchmark.
export interface CPUKernelConfig {registers: number; block: number; rowThreshold: number | null; unroll: 2 | 4}
export const CPU_CONFIG: Record<'128-simd' | '128-scalar' | '1024-simd' | '1024-scalar', CPUKernelConfig> = {
  '128-simd': {registers: 4, block: 32, rowThreshold: null, unroll: 4},
  '128-scalar': {registers: 8, block: 32, rowThreshold: null, unroll: 4},
  '1024-simd': {registers: 4, block: 32, rowThreshold: 512, unroll: 4},
  '1024-scalar': {registers: 8, block: 32, rowThreshold: null, unroll: 4},
};
