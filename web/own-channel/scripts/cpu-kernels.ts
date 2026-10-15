// Generate explicit register tiles for both handwritten CPU variants. Each
// output keeps its original input order, including the original f32 rounding.
import type {CPUKernelConfig} from './cpu-config.ts';
import {ROW_SLOTS} from '../src/cpu-memory.ts';
export function matrixKernels(simd: boolean, width: number, options: CPUKernelConfig = {registers: simd ? 4 : 8, block: 32, rowThreshold: simd && width === 1024 ? 512 : null, unroll: 4, rows: 1}): string {
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
${tile(simd || width % 4 !== 0 ? 1 : 4, 'matrix_tail')}
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
  // max(+0, z) as a mask that clears z <= 0: -0 becomes +0 and NaN passes.
  // V8 lowers the NaN-aware f32 max to a long (SIMD) or branchy (scalar) sequence.
  const relu = simd
    ? `(v128.andnot (local.tee $value (v128.load ${value})) (f32x4.le (local.get $value) (v128.const f32x4 0 0 0 0)))`
    : `(f32.reinterpret_i32 (i32.and (i32.reinterpret_f32 (local.tee $value (f32.load ${value}))) (i32.add (f32.le (local.get $value) (f32.const 0)) (i32.const -1))))`;
  return `
  (func $relu_add (param $dst i32) (param $skip i32) (param $z i32)
    (local $at i32) (local $value ${simd ? 'v128' : 'f32'})
    (loop $loop
      (${simd ? 'v128.store' : 'f32.store'} (i32.add (local.get $dst) (local.get $at))
        (${simd ? 'f32x4.add' : 'f32.add'} (${simd ? 'v128.load' : 'f32.load'} (i32.add (local.get $skip) (local.get $at))) ${relu}))
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

// Train and evaluate several rows per pass, so every matrix value loaded from
// memory serves all of them. Each output keeps its single-row summation order
// and each gradient element receives the rows in batch order: results stay
// bit-identical. The single-row kernels skip zero inputs, which only changes
// results for nonfinite weights, so a group with an exact zero input uses them.
export function rowKernels(simd: boolean, width: number, rows: number, block = width): string {
  if (rows === 1) return `
  (func $gradient_groups (param $rows i32) (param $scale f32) (result i32) (i32.const 0))
  (func $evaluate_groups (param $first i32) (param $end i32) (result i32) (local.get $first))`;
  if (![2, 4].includes(rows) || rows > ROW_SLOTS) throw new Error('Invalid CPU row group.');
  const type = simd ? 'v128' : 'f32', lanes = simd ? 4 : 1, step = 4 * lanes;
  const load = simd ? 'v128.load' : 'f32.load', store = simd ? 'v128.store' : 'f32.store';
  const add = simd ? 'f32x4.add' : 'f32.add', mul = simd ? 'f32x4.mul' : 'f32.mul';
  const splat = simd ? 'v128.load32_splat' : 'f32.load', zero = simd ? '(v128.const i32x4 0 0 0 0)' : '(f32.const 0)';
  // Row strides: states and normalized hold the input and hidden vectors.
  const vector = width * 4, pair = 2 * vector, dense = 97 * vector, logits = 400;
  // Eight accumulators: columns per forward tile, matrix rows per dot tile.
  const tile = 8 / rows;
  const each = (count: number, text: (i: number) => string, separator = '\n    ') => Array.from({length: count}, (_, i) => text(i)).join(separator);
  const at = (name: string, offset: number) => offset ? `(i32.add (global.get $${name}) (i32.const ${offset}))` : `(global.get $${name})`;
  const reduce = (sum: string) => simd
    ? `(f32.add (f32.add (f32x4.extract_lane 0 (local.get ${sum})) (f32x4.extract_lane 1 (local.get ${sum}))) (f32.add (f32x4.extract_lane 2 (local.get ${sum})) (f32x4.extract_lane 3 (local.get ${sum}))))`
    : `(local.get ${sum})`;

  // dst_r[o] = sum over ascending i of M[i][o] x_r[i], as $matrix_tile adds it.
  // With fewer inputs per block than the width, a column tile's matrix rows
  // stay in L1 for its neighbor, and partial sums wait in dst between blocks.
  const forward = (name: string, outputs: number, dstStride: number, block: number) => {
    const vectors = outputs / lanes, full = Math.floor(vectors / tile) * tile, blocked = block < width;
    const body = (columns: number, label: string) => `
    ${each(rows * columns, k => `(local.set $s${k} ${blocked ? `(${load} offset=${Math.floor(k / columns) * dstStride + k % columns * step} (i32.add (local.get $dst) (local.get $column)))` : zero})`, ' ')}
    (local.set $x (local.get $first))
    (local.set $a (i32.add (local.get $matrix) (i32.add (i32.mul (i32.sub (local.get $first) (local.get $input)) (i32.const ${outputs})) (local.get $column))))
    (loop $${label}
      ${each(columns, c => `(local.set $m${c} (${load} offset=${c * step} (local.get $a)))`, ' ')}
      ${each(rows, r => `(local.set $factor (${splat} offset=${r * pair} (local.get $x)))
      ${each(columns, c => `(local.set $s${r * columns + c} (${add} (local.get $s${r * columns + c}) (${mul} (local.get $m${c}) (local.get $factor))))`, '\n      ')}`, '\n      ')}
      (local.set $a (i32.add (local.get $a) (i32.const ${outputs * 4})))
      (local.set $x (i32.add (local.get $x) (i32.const 4)))
      (br_if $${label} (i32.lt_u (local.get $x) (local.get $last))))
    ${each(rows * columns, k => `(${store} offset=${Math.floor(k / columns) * dstStride + k % columns * step} (i32.add (local.get $dst) (local.get $column)) (local.get $s${k}))`)}`;
    const pass = `
    (local.set $last (i32.add (local.get $first) (i32.const ${(blocked ? block : width) * 4})))
    (local.set $column (i32.const 0))${full ? `
    (loop $tiles${body(tile, 'inputs')}
      (local.set $column (i32.add (local.get $column) (i32.const ${tile * step})))
      (br_if $tiles (i32.lt_u (local.get $column) (i32.const ${full * step}))))` : ''}${vectors > full ? body(vectors - full, 'tail') : ''}`;
    return `
  (func $${name} (param $dst i32) (param $input i32) (param $matrix i32)
    (local $column i32) (local $x i32) (local $first i32) (local $last i32) (local $a i32) (local $factor ${type})
    ${each(tile, c => `(local $m${c} ${type})`, ' ')}
    ${each(rows * tile, k => `(local $s${k} ${type})`, ' ')}
    (local.set $first (local.get $input))${blocked ? `
    ${each(rows, r => `(call $clear (i32.add (local.get $dst) (i32.const ${r * dstStride})) (i32.const ${outputs}))`, ' ')}
    (loop $blocks${pass}
      (local.set $first (local.get $last))
      (br_if $blocks (i32.lt_u (local.get $first) (i32.add (local.get $input) (i32.const ${vector})))))` : pass})`;
  };

  // dst_r[i] = $dot(M[i], dy_r): the same lane sums and final pairing.
  const dots = (name: string, outputs: number, dyStride: number) => `
  (func $${name} (param $dst i32) (param $dy i32) (param $matrix i32)
    (local $i i32) (local $m i32) (local $y i32) (local $yend i32) (local $value ${type})
    ${each(tile, k => `(local $m${k} ${type})`, ' ')}
    ${each(rows * tile, k => `(local $s${k} ${type})`, ' ')}
    (local.set $m (local.get $matrix))
    (loop $matrix_rows
      ${each(rows * tile, k => `(local.set $s${k} ${zero})`, ' ')}
      (local.set $y (local.get $dy))
      (local.set $yend (i32.add (local.get $dy) (i32.const ${outputs * 4})))
      (loop $chunks
        ${each(tile, k => `(local.set $m${k} (${load} offset=${k * outputs * 4} (local.get $m)))`, ' ')}
        ${each(rows, r => `(local.set $value (${load} offset=${r * dyStride} (local.get $y)))
        ${each(tile, k => `(local.set $s${k * rows + r} (${add} (local.get $s${k * rows + r}) (${mul} (local.get $m${k}) (local.get $value))))`, '\n        ')}`, '\n        ')}
        (local.set $m (i32.add (local.get $m) (i32.const ${step})))
        (local.set $y (i32.add (local.get $y) (i32.const ${step})))
        (br_if $chunks (i32.lt_u (local.get $y) (local.get $yend))))
      (local.set $m (i32.add (local.get $m) (i32.const ${(tile - 1) * outputs * 4})))
      ${each(rows * tile, k => `(f32.store offset=${k % rows * vector + Math.floor(k / rows) * 4} (i32.add (local.get $dst) (local.get $i)) ${reduce(`$s${k}`)})`, '\n      ')}
      (local.set $i (i32.add (local.get $i) (i32.const ${tile * 4})))
      (br_if $matrix_rows (i32.lt_u (local.get $i) (i32.const ${vector})))))`;

  // G[i][o] += dy_r[o] x_r[i] over ascending r, as successive $axpy calls add
  // it. Two gradient rows per pass share each dy load.
  const outer = (name: string, outputs: number, dyStride: number) => `
  (func $${name} (param $gradient i32) (param $x i32) (param $dy i32)
    (local $g i32) (local $end i32) (local $y i32) (local $xend i32) (local $value ${type}) (local $sum0 ${type}) (local $sum1 ${type})
    ${each(2 * rows, k => `(local $f${k} ${type})`, ' ')}
    (local.set $g (local.get $gradient))
    (local.set $xend (i32.add (local.get $x) (i32.const ${vector})))
    (loop $inputs
      ${each(2 * rows, k => `(local.set $f${k} (${splat} offset=${(k % rows) * pair + Math.floor(k / rows) * 4} (local.get $x)))`, ' ')}
      (local.set $y (local.get $dy))
      (local.set $end (i32.add (local.get $g) (i32.const ${outputs * 4})))
      (loop $chunks
        (local.set $sum0 (${load} (local.get $g)))
        (local.set $sum1 (${load} offset=${outputs * 4} (local.get $g)))
        ${each(rows, r => `(local.set $value (${load} offset=${r * dyStride} (local.get $y)))
        (local.set $sum0 (${add} (local.get $sum0) (${mul} (local.get $value) (local.get $f${r}))))
        (local.set $sum1 (${add} (local.get $sum1) (${mul} (local.get $value) (local.get $f${rows + r}))))`, '\n        ')}
        (${store} (local.get $g) (local.get $sum0))
        (${store} offset=${outputs * 4} (local.get $g) (local.get $sum1))
        (local.set $g (i32.add (local.get $g) (i32.const ${step})))
        (local.set $y (i32.add (local.get $y) (i32.const ${step})))
        (br_if $chunks (i32.lt_u (local.get $g) (local.get $end))))
      (local.set $g (i32.add (local.get $g) (i32.const ${outputs * 4})))
      (local.set $x (i32.add (local.get $x) (i32.const 8)))
      (br_if $inputs (i32.lt_u (local.get $x) (local.get $xend)))))`;

  // $norm and $norm_back of every row; their dot products run side by side.
  const norms = `
  (func $norm_rows (param $x i32) (param $dst i32) (param $inverse i32)
    (local $at i32) (local $value ${type}) (local $r f32) ${each(rows, r => `(local $s${r} ${type})`, ' ')}
    (loop $elements
      ${each(rows, r => `(local.set $value (${load} offset=${r * pair} (i32.add (local.get $x) (local.get $at))))
      (local.set $s${r} (${add} (local.get $s${r}) (${mul} (local.get $value) (local.get $value))))`, '\n      ')}
      (local.set $at (i32.add (local.get $at) (i32.const ${step})))
      (br_if $elements (i32.lt_u (local.get $at) (i32.const ${vector}))))
    ${each(rows, r => `(local.set $r (f32.div (f32.const 1) (f32.sqrt (f32.add (f32.div ${reduce(`$s${r}`)} (f32.const ${width})) (f32.const 0.00001)))))
    (f32.store offset=${16 * r} (local.get $inverse) (local.get $r))
    (memory.copy (i32.add (local.get $dst) (i32.const ${r * pair})) (i32.add (local.get $x) (i32.const ${r * pair})) (i32.const ${vector}))
    (call $multiply (i32.add (local.get $dst) (i32.const ${r * pair})) (i32.const ${width}) (local.get $r))`)})
  (func $norm_back_rows (param $normalized i32) (param $dy i32) (param $dx i32) (param $inverse i32)
    (local $at i32) ${each(rows, r => `(local $s${r} ${type})`, ' ')}
    (loop $elements
      ${each(rows, r => `(local.set $s${r} (${add} (local.get $s${r}) (${mul} (${load} offset=${r * pair} (i32.add (local.get $normalized) (local.get $at))) (${load} offset=${r * vector} (i32.add (local.get $dy) (local.get $at))))))`, '\n      ')}
      (local.set $at (i32.add (local.get $at) (i32.const ${step})))
      (br_if $elements (i32.lt_u (local.get $at) (i32.const ${vector}))))
    ${each(rows, r => `(memory.copy (i32.add (local.get $dx) (i32.const ${r * vector})) (i32.add (local.get $dy) (i32.const ${r * vector})) (i32.const ${vector}))
    (call $axpy (i32.add (local.get $dx) (i32.const ${r * vector})) (i32.add (local.get $normalized) (i32.const ${r * pair})) (i32.const ${width}) (f32.neg (f32.div ${reduce(`$s${r}`)} (f32.const ${width}))))
    (call $multiply (i32.add (local.get $dx) (i32.const ${r * vector})) (i32.const ${width}) (f32.load offset=${16 * r} (local.get $inverse)))`)})`;

  // Softmax of every row as $softmax computes it. The denominators of the
  // group accumulate side by side, each in its own ascending order.
  const softmax = (record: boolean) => `
  (func ${record ? '$record_rows (param $id i32)' : `$softmax_rows ${each(rows, r => `(param $t${r} i32)`, ' ')} (param $scale f32)`}
    (local $at i32) ${each(rows, r => `(local $m${r} f32) (local $w${r} i32) (local $d${r} f32)${record ? ` (local $t${r} i32)` : ''}`, ' ')}
    ${each(rows, r => `${record ? `(local.set $t${r} (i32.load offset=${12 * r + 8} (i32.add (global.get $data) (i32.mul (local.get $id) (i32.const 12)))))
    ` : ''}(if (i32.ge_u (local.get $t${r}) (i32.const 97)) (then unreachable))
    (local.set $m${r} (call $maximum (i32.const ${r * logits})))
    (local.set $w${r} (global.get $winner))
    (call $exponentials (local.get $m${r}) (i32.const ${r * logits}))`)}
    (local.set $at (global.get $errors))
    (loop $sums
      ${each(rows, r => `(local.set $d${r} (f32.add (local.get $d${r}) (f32.load offset=${r * logits} (local.get $at))))`, '\n      ')}
      (local.set $at (i32.add (local.get $at) (i32.const 4)))
      (br_if $sums (i32.lt_u (local.get $at) ${at('errors', 388)})))
    ${record ? each(rows, r => `(call $tally (i32.add (local.get $id) (i32.const ${r}))
      (f32.sub (f32.add (call $log (local.get $d${r})) (local.get $m${r})) (f32.load (i32.add ${at('logits', r * logits)} (i32.mul (local.get $t${r}) (i32.const 4)))))
      (i32.eq (local.get $w${r}) (local.get $t${r})))`) : `(if (f32.ne (local.get $scale) (f32.const 0))
      (then
        ${each(rows, r => `(call $multiply ${at('errors', r * logits)} (i32.const 100) (f32.div (local.get $scale) (local.get $d${r})))
        (f32.store (i32.add ${at('errors', r * logits)} (i32.mul (local.get $t${r}) (i32.const 4)))
          (f32.sub (f32.load (i32.add ${at('errors', r * logits)} (i32.mul (local.get $t${r}) (i32.const 4)))) (local.get $scale)))`, '\n        ')}))`})`;

  // Inputs and hidden vectors of the group, by their row slots.
  const ab = each(rows, r => `(local.get $a${r}) (local.get $b${r})`, ' ');
  return `
  (func $rows_have_zero (param $x i32) (result i32)
    (local $at i32) (local $found ${simd ? 'v128' : 'i32'})
    (loop $elements
      ${each(rows, r => `(local.set $found (${simd ? 'v128.or' : 'i32.or'} (local.get $found) (${simd ? 'f32x4.eq' : 'f32.eq'} (${load} offset=${r * pair} (i32.add (local.get $x) (local.get $at))) ${zero})))`, '\n      ')}
      (local.set $at (i32.add (local.get $at) (i32.const ${step})))
      (br_if $elements (i32.lt_u (local.get $at) (i32.const ${vector}))))
    ${simd ? '(v128.any_true (local.get $found))' : '(local.get $found)'})
${forward('rows_dense', width, vector, block)}
${forward('rows_readout', 100, logits, width)}
${dots('rows_dense_dots', width, vector)}
${dots('rows_readout_dots', 100, logits)}
${outer('rows_dense_outer', width, vector)}
${outer('rows_readout_outer', 100, logits)}
${norms}
${softmax(false)}
${softmax(true)}
  ;; Forward rows (a_r, b_r) into their slots. Bit 0 reports a zero input,
  ;; bit 1 a zero hidden vector.
  (func $forward_rows ${each(rows, r => `(param $a${r} i32) (param $b${r} i32)`, ' ')} (result i32)
    (local $zero i32)
    ${each(rows, r => `(if (i32.or (i32.ge_u (local.get $a${r}) (i32.const 97)) (i32.ge_u (local.get $b${r}) (i32.const 97))) (then unreachable))
    (memory.copy ${at('states', r * pair)} (i32.mul (local.get $a${r}) (i32.const ${vector})) (i32.const ${vector}))
    (call $rotary_add ${at('states', r * pair)} (i32.mul (local.get $b${r}) (i32.const ${vector})) (i32.const 0))`)}
    (call $norm_rows (global.get $states) (global.get $normalized) (global.get $inverse))
    (if (call $rows_have_zero (global.get $normalized))
      (then (local.set $zero (i32.const 1))
        ${each(rows, r => `(call $matrix ${at('branches', r * vector)} ${at('normalized', r * pair)} (i32.const ${dense}) (i32.const ${width}) (i32.const ${width}))`, '\n        ')})
      (else (call $rows_dense (global.get $branches) (global.get $normalized) (i32.const ${dense}))))
    ${each(rows, r => `(call $relu_add ${at('states', r * pair + vector)} ${at('states', r * pair)} ${at('branches', r * vector)})`)}
    (call $norm_rows ${at('states', vector)} ${at('normalized', vector)} ${at('inverse', 4)})
    (if (call $rows_have_zero ${at('normalized', vector)})
      (then (local.set $zero (i32.or (local.get $zero) (i32.const 2)))
        ${each(rows, r => `(call $matrix ${at('logits', r * logits)} ${at('normalized', r * pair + vector)} (global.get $readout) (i32.const ${width}) (i32.const 100))`, '\n        ')})
      (else (call $rows_readout (global.get $logits) ${at('normalized', vector)} (global.get $readout))))
    (local.get $zero))
  (func $gradient_groups (param $rows i32) (param $scale f32) (result i32)
    (local $row i32) (local $id i32) (local $at i32) (local $zero i32)
    ${each(rows, r => `(local $a${r} i32) (local $b${r} i32) (local $t${r} i32)`, ' ')}
    (block $done
      (loop $groups
        (br_if $done (i32.gt_u (i32.add (local.get $row) (i32.const ${rows})) (local.get $rows)))
        ${each(rows, r => `(local.set $id (i32.load offset=${4 * r} (i32.add (global.get $samples) (i32.mul (local.get $row) (i32.const 4)))))
        (if (i32.ge_u (local.get $id) (i32.const 4656)) (then unreachable))
        (local.set $at (i32.add (global.get $data) (i32.mul (local.get $id) (i32.const 12))))
        (local.set $a${r} (i32.load (local.get $at))) (local.set $b${r} (i32.load offset=4 (local.get $at))) (local.set $t${r} (i32.load offset=8 (local.get $at)))`, '\n        ')}
        (local.set $zero (call $forward_rows ${ab}))
        (call $softmax_rows ${each(rows, r => `(local.get $t${r})`, ' ')} (local.get $scale))
        (if (i32.and (local.get $zero) (i32.const 2))
          (then
            ${each(rows, r => `(call $matrix_back (global.get $readout_gradient) ${at('dn', r * vector)} ${at('normalized', r * pair + vector)} ${at('errors', r * logits)} (global.get $readout) (i32.const ${width}) (i32.const 100))`, '\n            ')})
          (else
            (call $rows_readout_dots (global.get $dn) (global.get $errors) (global.get $readout))
            (call $rows_readout_outer (global.get $readout_gradient) ${at('normalized', vector)} (global.get $errors))))
        (call $norm_back_rows ${at('normalized', vector)} (global.get $dn) (global.get $dh) ${at('inverse', 4)})
        ${each(rows, r => `(call $relu_back ${at('dn', r * vector)} ${at('branches', r * vector)} ${at('dh', r * vector)})`, '\n        ')}
        (if (i32.and (local.get $zero) (i32.const 1))
          (then
            ${each(rows, r => `(call $matrix_back ${at('gradient', dense)} ${at('temporary', r * vector)} ${at('normalized', r * pair)} ${at('dn', r * vector)} (i32.const ${dense}) (i32.const ${width}) (i32.const ${width}))`, '\n            ')})
          (else
            (call $rows_dense_dots (global.get $temporary) (global.get $dn) (i32.const ${dense}))
            (call $rows_dense_outer ${at('gradient', dense)} (global.get $normalized) (global.get $dn))))
        (call $norm_back_rows (global.get $normalized) (global.get $temporary) (global.get $dn) (global.get $inverse))
        ${each(rows, r => `(call $axpy ${at('dh', r * vector)} ${at('dn', r * vector)} (i32.const ${width}) (f32.const 1))
        (call $axpy (i32.add (global.get $gradient) (i32.mul (local.get $a${r}) (i32.const ${vector}))) ${at('dh', r * vector)} (i32.const ${width}) (f32.const 1))
        (call $rotary_add (i32.add (global.get $gradient) (i32.mul (local.get $b${r}) (i32.const ${vector}))) ${at('dh', r * vector)} (i32.const 1))`, '\n        ')}
        (local.set $row (i32.add (local.get $row) (i32.const ${rows})))
        (br $groups)))
    (local.get $row))
  (func $evaluate_groups (param $first i32) (param $end i32) (result i32)
    (local $id i32) (local $at i32)
    ${each(rows, r => `(local $a${r} i32) (local $b${r} i32)`, ' ')}
    (local.set $id (local.get $first))
    (block $done
      (loop $groups
        (br_if $done (i32.gt_u (i32.add (local.get $id) (i32.const ${rows})) (local.get $end)))
        (local.set $at (i32.add (global.get $data) (i32.mul (local.get $id) (i32.const 12))))
        ${each(rows, r => `(local.set $a${r} (i32.load offset=${12 * r} (local.get $at))) (local.set $b${r} (i32.load offset=${12 * r + 4} (local.get $at)))`, '\n        ')}
        (drop (call $forward_rows ${ab}))
        (call $record_rows (local.get $id))
        (local.set $id (i32.add (local.get $id) (i32.const ${rows})))
        (br $groups)))
    (local.get $id))`;
}
