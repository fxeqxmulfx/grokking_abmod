import {writeFile} from 'node:fs/promises';
import {root} from '../project.ts';
import {CPU_CONFIG} from './cpu-config.ts';
import {assembleCPU} from './assemble-cpu.ts';

for (const [model, width] of [['dense-rotary', 128], ['dense-rotary-wide', 1024]] as const) {
  for (const variant of ['simd', 'scalar'] as const) for (const shared of [false, true]) {
    const name = `${model}-${variant}${shared ? '-shared' : ''}`;
    const bytes = await assembleCPU(model, variant, CPU_CONFIG[`${width}-${variant}`], shared);
    await writeFile(new URL(`dist/${name}.wasm`, root), bytes);
    console.log(`${name}: ${bytes.length.toLocaleString('en-US')} bytes`);
  }
}
