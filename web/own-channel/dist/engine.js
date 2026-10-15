import { BACKENDS, errorText } from './types.js';
import { modelInfo, trainingOptimizer } from './models.js';
import { WASMTrainer } from './wasm.js';
import { ParallelWASMTrainer } from './parallel-wasm.js';
export async function createEngine(seed, backend = 'auto', model = 'dense-rotary', preset = 'fast', custom) {
    if (!BACKENDS.includes(backend))
        throw new Error('Unknown training engine.');
    const { learningRate, weightDecay: decay } = trainingOptimizer(model, preset, custom);
    const scalarWidth = modelInfo(model).width % 4 !== 0;
    const scalar = backend.startsWith('scalar') || scalarWidth, requireSIMD = backend.startsWith('simd') && !scalarWidth;
    const available = globalThis.crossOriginIsolated && typeof SharedArrayBuffer === 'function' && typeof Worker === 'function';
    const cores = typeof navigator === 'object' ? navigator.hardwareConcurrency || 1 : 1;
    // Four training threads win on the measured six-core CPU. Reserve capacity
    // for the UI/render worker; smaller devices use two or one training thread.
    const requested = backend === 'auto' ? (cores >= 6 ? 4 : cores >= 4 ? 2 : 1) : Number(backend.split('-')[1] ?? 1);
    let note = scalarWidth && !backend.startsWith('scalar') ? 'This width uses scalar WASM; SIMD requires a multiple of four channels.' : undefined;
    const describe = (engine) => {
        if (note) {
            const info = engine.info();
            engine.info = () => ({ ...info, note });
        }
        return engine;
    };
    if (requested > 1 && available) {
        try {
            return describe(await ParallelWASMTrainer.create(seed, { model, scalar, requireSIMD, threads: requested, learningRate, decay }));
        }
        catch (error) {
            note = [note, `Parallel WASM could not start; using one CPU thread. ${errorText(error)}`].filter(Boolean).join(' ');
        }
    }
    else if (requested > 1)
        note = [note, 'Shared memory is unavailable; using one CPU thread.'].filter(Boolean).join(' ');
    const engine = await WASMTrainer.create(seed, { model, scalar, requireSIMD, learningRate, decay });
    return describe(engine);
}
//# sourceMappingURL=engine.js.map