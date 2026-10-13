// Generate explicit register tiles for both handwritten CPU variants. Each
// output keeps its original input order, including the original f32 rounding.
import type {CPUKernelConfig} from './cpu-config.ts';
export function matrixKernels(simd: boolean, width: number, options: CPUKernelConfig = {registers: simd ? 4 : 8, block: 32, rowThreshold: simd && width === 1024 ? 512 : null, unroll: 4}): string {
  if (![2, 4, 8, 16].includes(options.registers) || (!simd && options.registers < 4) || ![16, 32, 64, 128].includes(options.block)) throw new Error('Invalid CPU kernel configuration.');
  const type = simd ? 'v128' : 'f32', lanes = simd ? 4 : 1;
  const load = simd ? 'v128.load' : 'f32.load', store = simd ? 'v128.store' : 'f32.store';
  const add = simd ? 'f32x4.add' : 'f32.add', mul = simd ? 'f32x4.mul' : 'f32.mul';
  const tile = (registers: number, name: string) => `
  (func $${name} (param $dst i32) (param $input i32) (param $matrix i32) (param $inputs i32) (param $outputs i32) (param $first i32) (param $end i32)
    (local $block i32) (local $stop i32) (local $column i32) (local $x i32) (local $xend i32) (local $a i32) (local $out i32) (local $stride i32)
    (local $value f32) (local $factor ${type})
    ${Array.from({length: registers}, (_, i) => `(local $sum${i} ${type})`).join(' ')}
    (local.set $stride (i32.mul (local.get $outputs) (i32.const 4)))
    (loop $blocks
      (local.set $stop (i32.add (local.get $block) (i32.const ${options.block})))
      (if (i32.gt_u (local.get $stop) (local.get $inputs)) (then (local.set $stop (local.get $inputs))))
      (local.set $column (local.get $first))
      (loop $columns
        (local.set $out (i32.add (local.get $dst) (i32.mul (local.get $column) (i32.const 4))))
        ${Array.from({length: registers}, (_, i) => `(local.set $sum${i} (${load} offset=${i * lanes * 4} (local.get $out)))`).join('\n        ')}
        (local.set $x (i32.add (local.get $input) (i32.mul (local.get $block) (i32.const 4))))
        (local.set $xend (i32.add (local.get $input) (i32.mul (local.get $stop) (i32.const 4))))
        (local.set $a (i32.add (local.get $matrix) (i32.add (i32.mul (local.get $block) (local.get $stride)) (i32.mul (local.get $column) (i32.const 4)))))
        (loop $rows
          (local.set $value (f32.load (local.get $x)))
          (if (f32.ne (local.get $value) (f32.const 0)) (then
            (local.set $factor ${simd ? '(f32x4.splat (local.get $value))' : '(local.get $value)'})
            ${Array.from({length: registers}, (_, i) => `(local.set $sum${i} (${add} (local.get $sum${i}) (${mul} (${load} offset=${i * lanes * 4} (local.get $a)) (local.get $factor))))`).join('\n            ')}))
          (local.set $x (i32.add (local.get $x) (i32.const 4)))
          (local.set $a (i32.add (local.get $a) (local.get $stride)))
          (br_if $rows (i32.lt_u (local.get $x) (local.get $xend))))
        ${Array.from({length: registers}, (_, i) => `(${store} offset=${i * lanes * 4} (local.get $out) (local.get $sum${i}))`).join('\n        ')}
        (local.set $column (i32.add (local.get $column) (i32.const ${registers * lanes})))
        (br_if $columns (i32.lt_u (local.get $column) (local.get $end))))
      (local.set $block (local.get $stop))
      (br_if $blocks (i32.lt_u (local.get $block) (local.get $inputs)))))`;
  const registers = options.registers, columns = registers * lanes;
  const rowMatrix = `
  (func $matrix_rows (param $dst i32) (param $input i32) (param $matrix i32) (param $inputs i32) (param $outputs i32)
    (local $i i32) (local $value f32)
    (call $clear (local.get $dst) (local.get $outputs))
    (loop $rows
      (local.set $value (f32.load (i32.add (local.get $input) (i32.mul (local.get $i) (i32.const 4)))))
      (if (f32.ne (local.get $value) (f32.const 0)) (then
        (call $axpy (local.get $dst) (i32.add (local.get $matrix) (i32.mul (i32.mul (local.get $i) (local.get $outputs)) (i32.const 4))) (local.get $outputs) (local.get $value))))
      (local.set $i (i32.add (local.get $i) (i32.const 1))) (br_if $rows (i32.lt_u (local.get $i) (local.get $inputs)))))`;
  return `${options.rowThreshold !== null ? rowMatrix : ''}
${tile(registers, 'matrix_tile')}
${tile(simd ? 1 : 4, 'matrix_tail')}
  (func $matrix (param $dst i32) (param $input i32) (param $matrix i32) (param $inputs i32) (param $outputs i32)
    (local $end i32)
    ${options.rowThreshold !== null ? `(if (i32.ge_u (local.get $outputs) (i32.const ${options.rowThreshold})) (then (call $matrix_rows (local.get $dst) (local.get $input) (local.get $matrix) (local.get $inputs) (local.get $outputs)) (return)))` : ''}
    (call $clear (local.get $dst) (local.get $outputs))
    (local.set $end (i32.and (local.get $outputs) (i32.const ${-columns})))
    (if (local.get $end) (then (call $matrix_tile (local.get $dst) (local.get $input) (local.get $matrix) (local.get $inputs) (local.get $outputs) (i32.const 0) (local.get $end))))
    (if (i32.lt_u (local.get $end) (local.get $outputs)) (then (call $matrix_tail (local.get $dst) (local.get $input) (local.get $matrix) (local.get $inputs) (local.get $outputs) (local.get $end) (local.get $outputs)))))`;
}

export function reluKernels(simd: boolean, width: number): string {
  const value = '(i32.add (local.get $z) (local.get $at))';
  return `
  (func $relu_add (param $dst i32) (param $skip i32) (param $z i32)
    (local $at i32)
    (loop $loop
      (${simd ? 'v128.store' : 'f32.store'} (i32.add (local.get $dst) (local.get $at))
        (${simd ? 'f32x4.add' : 'f32.add'} (${simd ? 'v128.load' : 'f32.load'} (i32.add (local.get $skip) (local.get $at)))
          (${simd ? 'f32x4.max' : 'f32.max'} ${simd ? '(v128.const f32x4 0 0 0 0)' : '(f32.const 0)'} (${simd ? 'v128.load' : 'f32.load'} ${value}))))
      (local.set $at (i32.add (local.get $at) (i32.const ${simd ? 16 : 4})))
      (br_if $loop (i32.lt_u (local.get $at) (i32.const ${width * 4})))))
  (func $relu_back (param $dst i32) (param $z i32) (param $dy i32)
    (local $at i32)
    (loop $loop
      (${simd ? 'v128.store' : 'f32.store'} (i32.add (local.get $dst) (local.get $at))
        (${simd ? 'f32x4.mul' : 'f32.mul'} (${simd ? 'v128.load' : 'f32.load'} (i32.add (local.get $dy) (local.get $at)))
          ${simd ? `(f32x4.convert_i32x4_u (i32x4.shr_u (f32x4.gt (v128.load ${value}) (v128.const f32x4 0 0 0 0)) (i32.const 31)))` : `(f32.convert_i32_u (f32.gt (f32.load ${value}) (f32.const 0)))`}))
      (local.set $at (i32.add (local.get $at) (i32.const ${simd ? 16 : 4})))
      (br_if $loop (i32.lt_u (local.get $at) (i32.const ${width * 4})))))`;
}

export function unrollVectorKernels(source: string, simd: boolean, repetitions: 2 | 4 = 4): string {
  // Preserve a single accumulator in dot: unrolling does not reassociate sums.
  for (const name of ['clear', 'axpy', 'dot', 'multiply']) {
    const start = source.indexOf(`(func $${name} `), next = source.indexOf('\n(func ', start + 1);
    const end = next < 0 ? source.length : next;
    let fn = source.slice(start, end);
    const loop = fn.indexOf('  (loop $loop'), bodyStart = fn.indexOf('\n', loop) + 1;
    const advance = fn.indexOf('    (local.set $', bodyStart);
    // Dot starts by setting its accumulator, which belongs to the arithmetic body.
    const bodyEnd = fn.indexOf('    (local.set $b (i32.add', bodyStart);
    const last = name === 'dot' ? bodyEnd : advance;
    const body = fn.slice(bodyStart, last), stride = simd ? 16 : 4;
    const pointers = name === 'dot' ? ['a', 'b'] : name === 'axpy' ? ['dst', 'src'] : ['dst'];
    const expanded = Array.from({length: repetitions}, (_, i) => {
      let text = body;
      if (i) for (const pointer of pointers) text = text.replaceAll(`(local.get $${pointer})`, `(i32.add (local.get $${pointer}) (i32.const ${stride * i}))`);
      return text;
    }).join('');
    // The scalar lengths are divisible by four; SIMD handles the four-lane tail.
    if (simd) {
      const suffixStart = fn.indexOf('\n  (f32.add ', last);
      const suffix = suffixStart < 0 ? ')' : fn.slice(suffixStart);
      const tailBody = body + pointers.map(pointer => `    (local.set $${pointer} (i32.add (local.get $${pointer}) (i32.const ${stride})))\n`).join('');
      const advances = pointers.map(pointer => `    (local.set $${pointer} (i32.add (local.get $${pointer}) (i32.const ${stride * repetitions})))\n`).join('');
      const head = fn.slice(0, loop);
      const boundPointer = name === 'dot' ? 'a' : 'dst';
      fn = `${head}  (block $tail\n    (loop $unrolled\n      (br_if $tail (i32.gt_u (i32.add (local.get $${boundPointer}) (i32.const ${stride * repetitions})) (local.get $end)))\n${expanded}${advances}      (br $unrolled)))\n  (if (i32.lt_u (local.get $${boundPointer}) (local.get $end)) (then\n    (loop $remaining\n${tailBody}      (br_if $remaining (i32.lt_u (local.get $${boundPointer}) (local.get $end))))))${suffix}`;
    } else {
      fn = fn.slice(0, bodyStart) + expanded + fn.slice(last).replaceAll('(i32.const 4)', `(i32.const ${4 * repetitions})`);
    }
    source = source.slice(0, start) + fn + source.slice(end);
  }
  return source;
}
