export const MODEL_IDS = ['dense-rotary-16', 'dense-rotary-32', 'dense-rotary', 'dense-rotary-wide'];
export const BACKENDS = ['auto', 'simd', 'scalar', 'simd-2', 'simd-4', 'scalar-2', 'scalar-4'];
export const errorText = (error) => error instanceof Error ? error.message : String(error);
export function send(worker, message) { worker?.postMessage(message); }
export function backend(value) {
    if (!BACKENDS.includes(value))
        throw new Error('Unknown training engine.');
    return value;
}
//# sourceMappingURL=types.js.map