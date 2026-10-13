import wabtFactory from 'wabt';
import {readFile, writeFile} from 'node:fs/promises';
import {root} from '../project.ts';
const wabt = await wabtFactory(), source = await readFile(new URL('wasm/topology.wat', root), 'utf8');
for (const variant of ['simd', 'scalar'] as const) {
  const simd = variant === 'simd';
  const functions = ['add', 'subtract'].map(name => {
    const operation = name === 'add' ? 'add' : 'sub';
    return `(func $${name} (param $dst i32) (param $src i32)\n${simd
      ? `  (v128.store (local.get $dst) (f32x4.${operation} (v128.load (local.get $dst)) (v128.load (local.get $src))))`
      : [0, 4, 8, 12].map(offset => `  (f32.store offset=${offset} (local.get $dst) (f32.${operation} (f32.load offset=${offset} (local.get $dst)) (f32.load offset=${offset} (local.get $src))))`).join('\n')}\n)`;
  }).join('\n');
  const module = wabt.parseWat(`topology-${variant}.wat`, source.replace(';; @RASTER_VECTOR', functions), {simd});
  try {
    module.resolveNames(); module.validate(); const bytes = new Uint8Array(module.toBinary({canonicalize_lebs: true, write_debug_names: false}).buffer);
    if (!WebAssembly.validate(bytes)) throw new Error('Invalid CPU rasterizer.');
    await writeFile(new URL(`dist/topology-${variant}.wasm`, root), bytes); console.log(`topology-${variant}: ${bytes.length} bytes`);
  } finally {module.destroy();}
}
