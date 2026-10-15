import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {root} from '../project.ts';
import type {RasterExports} from '../dist/topology-scene.js';

for (const variant of ['simd', 'scalar']) test(`${variant} software raster: shared E roles, Rotate90 signs, dense edges and clipping`, async () => {
  const bytes = new Uint8Array(await readFile(new URL(`dist/topology-${variant}.wasm`, root)));
  const {instance} = await WebAssembly.instantiate(bytes), exports = instance.exports as unknown as RasterExports;
  exports.memory.grow(1); const buffer = exports.memory.buffer;
  const columns = 0, residues = 8, answers = residues + 194 * 4, palette = 2048, codes = palette + 4097 * 16;
  const scratch = codes + 400, pixels = scratch + 9 * 12 * 16;
  const column = new Float32Array(buffer, columns, 2), residue = new Float32Array(buffer, residues, 194), answer = new Float32Array(buffer, answers, 97);
  const color = new Float32Array(buffer, palette, 4097 * 4), weights = new Uint16Array(buffer, codes, 198), output = new Uint8Array(buffer, pixels, 8 * 12 * 4);
  column.set([2, 6]); residue.fill(2); residue[97] = 6; answer.fill(2);
  color.set([90, 0, 0, 1], 0); color.set([0, 0, 90, 1], 1024 * 4); color.set([0, 90, 0, 1], 2048 * 4);
  const render = (rows = [0, 4, 8, 12], height = 12) => exports.render(codes, palette, columns, residues, answers, scratch, pixels, 8, height, 2, rows[0], rows[1], rows[2], rows[3]);
  const pixel = (x: number, y: number) => Array.from(output.slice((y * 8 + x) * 4, (y * 8 + x + 1) * 4));
  weights.fill(65535); weights[0] = 0; render();
  assert.deepEqual(pixel(2, 0), [90, 0, 0, 127], 'a uses E[0,0]');
  assert.deepEqual(pixel(6, 0), [90, 0, 0, 127], 'odd Rotate90 output uses positive E[0,0]');
  assert.deepEqual(pixel(2, 8), [90, 0, 0, 127], 'readout uses the same E[0,0]');
  assert.deepEqual(pixel(4, 4), [0, 0, 0, 0], 'E does not create dense edges');
  weights.fill(65535); weights[1] = 0; render();
  assert.deepEqual(pixel(2, 0), [90, 0, 0, 127], 'a stays positive');
  assert.deepEqual(pixel(5, 0), [0, 0, 90, 127], 'even Rotate90 output negates E[0,1]');
  assert.deepEqual(pixel(5, 8), [90, 0, 0, 127], 'readout retains the original sign');
  weights.fill(65535); weights[195] = 2048; render();
  assert.deepEqual(pixel(2, 4), [0, 90, 0, 127], 'A[0,1] reaches output 1');
  assert.deepEqual(pixel(2, 0), [0, 0, 0, 0]); assert.deepEqual(pixel(5, 8), [0, 0, 0, 0]);
  render([-20, -16, -12, -8]); assert.ok(output.every(value => value === 0), 'offscreen rows are clipped');
  column.set([-100, 100]); render();
  assert.ok(Array.from({length: 8}, (_, x) => pixel(x, 6)[3]).every(alpha => alpha === 127), 'zoomed spans cross every visible column');
  weights.fill(65535); render(); assert.ok(output.every(value => value === 0), 'zero changes clear previous pixels');
  column.set([2, 6]); weights[0] = weights[1] = 0; weights[195] = 2048;
  render(); const complete = output.slice(), tiled = new Uint8Array(complete.length);
  for (let top = 0; top < 12; top += 3) {
    render([0, 4, 8, 12].map(row => row - top), 3);
    tiled.set(output.subarray(0, 8 * 3 * 4), top * 8 * 4);
  }
  assert.deepEqual(tiled, complete, 'Tiled accumulation preserves all shared-E roles, dense edges and strip boundaries.');
});
