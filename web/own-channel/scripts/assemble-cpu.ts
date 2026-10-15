import wabtFactory from 'wabt';
import {readFile} from 'node:fs/promises';
import {root} from '../project.ts';
import type {ModelId} from '../src/types.ts';
import {modelInfo} from '../src/models.ts';
import {denseWat, sharedWat} from './build-dense-trainers.ts';
import type {CPUKernelConfig} from './cpu-config.ts';
import {matrixKernels, reluKernels, rowKernels, unrollVectorKernels} from './cpu-kernels.ts';

const factory = wabtFactory();
const sources = new Map<string, Promise<string>>();
function read(name: string) {
  if (!sources.has(name)) sources.set(name, readFile(new URL(`wasm/${name}.wat`, root), 'utf8'));
  return sources.get(name)!;
}
export async function assembleCPU(model: ModelId, variant: 'simd' | 'scalar', config: CPUKernelConfig, shared = false): Promise<Uint8Array<ArrayBuffer>> {
  const width = modelInfo(model).width, simd = variant === 'simd', weightBytes = (97 * width + width * width) * 4;
  if (simd && width % 4 !== 0) throw new Error('SIMD requires a width divisible by four.');
  const [common, vectorSource, rotarySource, wabt] = await Promise.all([width === 128 ? read('dense-rotary') : denseWat(width), read(variant), read(`rotary-${variant}`), factory]);
  const vectors = vectorSource.replaceAll('@WEIGHT_BYTES', String(weightBytes));
  // Unrolled scalar loops require four-element lengths. Keep the single-lane
  // kernels for other even widths, and avoid partial blocks in grouped rows.
  const kernels = width % 4 === 0 ? unrollVectorKernels(vectors, simd, config.unroll) : vectors;
  const rotary = rotarySource.replaceAll('(i32.const 512)', `(i32.const ${width * 4})`);
  const source = common.replace(';; @VECTOR_KERNELS', kernels).replace(';; @ROTARY_KERNELS', rotary)
    .replace(';; @MATRIX_KERNELS', matrixKernels(simd, width, config)).replace(';; @RELU_KERNELS', reluKernels(simd, width))
    .replace(';; @ROW_KERNELS', rowKernels(simd, width, config.rows, width % config.block === 0 ? config.block : width));
  const name = `${model}-${variant}${shared ? '-shared' : ''}`, module = wabt.parseWat(`${name}.wat`, shared ? sharedWat(source, width) : source, {simd, threads: shared});
  try {
    module.resolveNames();
    // The runtime accepts feature flags; WABT's declaration omits the argument.
    (module.validate as (features: {simd: boolean; threads: boolean}) => void)({simd, threads: shared});
    const bytes = new Uint8Array(module.toBinary({canonicalize_lebs: true, write_debug_names: false}).buffer);
    if (!WebAssembly.validate(bytes)) throw new Error(`Invalid ${name} module.`);
    return bytes;
  } finally {module.destroy();}
}
