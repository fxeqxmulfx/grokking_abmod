export const PRIVATE_ADDRESSES = ['gradient', 'samples', 'readout', 'readout_gradient', 'states', 'normalized', 'branches', 'inverse', 'dh', 'dn', 'temporary', 'logits', 'errors', 'metrics', 'statistics', 'input', 'hidden'];
// Each activation buffer holds this many rows; the trainers address row r of a
// buffer at r times its single-row size.
export const ROW_SLOTS = 4;
export function cpuMemoryLayout(width, threads) {
    if (![16, 32, 64, 128, 256].includes(width) || !Number.isInteger(threads) || threads < 1 || threads > 8)
        throw new Error('Invalid CPU memory layout.');
    const weightBytes = (97 * width + width * width) * 4, vectorBytes = width * 4;
    const align = (value) => Math.ceil(value / 64) * 64;
    const control = align(3 * weightBytes + 9312 * 12);
    let cursor = control + threads * 64;
    const slots = [];
    for (let thread = 0; thread < threads; thread++) {
        const addresses = {};
        const allocate = (name, bytes) => { cursor = align(cursor); addresses[name] = cursor; cursor += bytes; };
        allocate('gradient', weightBytes);
        allocate('samples', 512 * 4);
        for (const name of ['readout', 'readout_gradient'])
            allocate(name, width * 100 * 4);
        allocate('states', ROW_SLOTS * 2 * vectorBytes);
        allocate('normalized', ROW_SLOTS * 2 * vectorBytes);
        allocate('branches', ROW_SLOTS * vectorBytes);
        allocate('inverse', ROW_SLOTS * 16);
        for (const name of ['dh', 'dn', 'temporary'])
            allocate(name, ROW_SLOTS * vectorBytes);
        allocate('logits', ROW_SLOTS * 400);
        allocate('errors', ROW_SLOTS * 400);
        allocate('metrics', 16);
        allocate('statistics', 32);
        addresses.input = addresses.states;
        addresses.hidden = addresses.states + vectorBytes;
        slots.push(addresses);
    }
    return { pages: Math.ceil(cursor / 65536), control, slots, weightBytes };
}
//# sourceMappingURL=cpu-memory.js.map