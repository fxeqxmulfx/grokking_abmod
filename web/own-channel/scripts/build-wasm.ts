import {writeFile} from 'node:fs/promises';
import {root} from '../project.ts';
import {CPU_CONFIG} from './cpu-config.ts';
import {assembleCPU} from './assemble-cpu.ts';
import {MODELS} from '../src/models.ts';
import {denseWat} from './build-dense-trainers.ts';

for (const {id: model, width} of Object.values(MODELS)) {
  if (width !== 128) await writeFile(new URL(`wasm/${model}.wat`, root), denseWat(width));
  for (const variant of ['simd', 'scalar'] as const) for (const shared of [false, true]) {
    const name = `${model}-${variant}${shared ? '-shared' : ''}`;
    // Narrower widths share the W128 kernel tuning.
    const bytes = await assembleCPU(model, variant, CPU_CONFIG[`${Math.max(width, 128)}-${variant}` as keyof typeof CPU_CONFIG], shared);
    await writeFile(new URL(`dist/${name}.wasm`, root), bytes);
    console.log(`${name}: ${bytes.length.toLocaleString('en-US')} bytes`);
  }
}
