export const MODEL_IDS = ['dense-rotary-16', 'dense-rotary-32', 'dense-rotary-64', 'dense-rotary', 'dense-rotary-256'];
// Optimizer presets: fast generalizes earliest on every native seed, plateau
// memorizes longest before generalizing, slow generalizes latest.
export const PRESETS = ['fast', 'plateau', 'slow'];
export const BACKENDS = ['auto', 'simd', 'scalar', 'simd-2', 'simd-4', 'scalar-2', 'scalar-4'];
export const errorText = (error) => error instanceof Error ? error.message : String(error);
export function send(worker, message) { worker?.postMessage(message); }
export function preset(value) {
    if (!PRESETS.includes(value))
        throw new Error('Unknown optimizer preset.');
    return value;
}
export function backend(value) {
    if (!BACKENDS.includes(value))
        throw new Error('Unknown training engine.');
    return value;
}
//# sourceMappingURL=types.js.map