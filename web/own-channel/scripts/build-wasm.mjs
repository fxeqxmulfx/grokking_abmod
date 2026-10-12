import wabtFactory from 'wabt';
import {readFile, writeFile} from 'node:fs/promises';

// WABT assembles the handwritten WAT. No source-language compiler is involved.
const wabt = await wabtFactory();
const root = new URL('../', import.meta.url);
for (const [trainer, weightBytes] of [['train', 75520], ['concat', 58624], ['residual', 336640], ['neighbor', 155136], ['neighbor-concat', 186880], ['neighbor-rotary', 55808], ['neighbor-rotary-wide', 446464]]) {
  const common = await readFile(new URL(`wasm/${trainer}.wat`, root), 'utf8');
  for (const variant of ['simd', 'scalar']) {
    const kernels = (await readFile(new URL(`wasm/${variant}.wat`, root), 'utf8')).replace('(i32.const 75520)', `(i32.const ${weightBytes})`);
    let source = common.replace(';; @VECTOR_KERNELS', kernels);
    if (trainer.startsWith('neighbor-rotary')) {
      const rotary = (await readFile(new URL(`wasm/rotary-${variant}.wat`, root), 'utf8')).replaceAll('(i32.const 512)', `(i32.const ${trainer.endsWith('-wide') ? 4096 : 512})`);
      source = source.replace(';; @ROTARY_KERNELS', rotary);
    }
    const module = wabt.parseWat(`${trainer}-${variant}.wat`, source, {simd: variant === 'simd'});
    try {
      module.resolveNames(); module.validate({simd: variant === 'simd'});
      const {buffer} = module.toBinary({canonicalize_lebs: true, write_debug_names: false});
      if (!WebAssembly.validate(buffer)) throw new Error(`Invalid ${variant} module`);
      await writeFile(new URL(`dist/${trainer}-${variant}.wasm`, root), buffer);
      console.log(`${trainer}-${variant}: ${buffer.length.toLocaleString('en-US')} bytes`);
    } finally { module.destroy(); }
  }
}
