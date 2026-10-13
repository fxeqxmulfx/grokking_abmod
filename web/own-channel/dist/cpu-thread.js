import { errorText } from './types.js';
let exports, memory, samples, control, slot;
onmessage = async (event) => {
    try {
        const request = event.data;
        if (request.type === 'initialize') {
            const instance = await WebAssembly.instantiate(request.module, { env: { memory: request.memory, ...request.addresses } });
            exports = instance.exports;
            memory = request.memory;
            slot = request.slot;
            samples = new Uint32Array(memory.buffer, request.addresses.samples, 512);
            control = new Int32Array(memory.buffer, request.control, (slot + 1) * 16);
            postMessage({ type: 'ready', job: 0 });
            return;
        }
        // Acquire the master's weights before refreshing the private readout.
        if (!exports || Atomics.load(control, 0) !== request.job)
            throw new Error('Invalid CPU worker phase.');
        exports.prepare();
        if (request.type === 'gradient') {
            samples.set(new Uint32Array(memory.buffer, request.source + request.first * 4, request.count));
            exports.gradient_batch_scaled(request.count, request.scale);
        }
        else
            exports.evaluate_range(request.first, request.end);
        // Release gradients/statistics before the master consumes this slot.
        Atomics.store(control, slot * 16, request.job);
        postMessage({ type: 'done', job: request.job });
    }
    catch (error) {
        postMessage({ type: 'error', message: errorText(error) });
    }
};
//# sourceMappingURL=cpu-thread.js.map