export declare const PRIVATE_ADDRESSES: readonly ['gradient', 'samples', 'readout', 'readout_gradient', 'states', 'normalized', 'branches', 'inverse', 'dh', 'dn', 'temporary', 'logits', 'errors', 'metrics', 'statistics', 'input', 'hidden'];
export type CPUAddresses = Record<typeof PRIVATE_ADDRESSES[number], number>;
export interface SharedCPU {
    memory: WebAssembly.Memory;
    addresses: CPUAddresses;
}
export declare function cpuMemoryLayout(width: number, threads: number): {
    pages: number;
    control: number;
    slots: CPUAddresses[];
    weightBytes: number;
};
