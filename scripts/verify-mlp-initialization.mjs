// Verify native CLI initialization with paired matrices and first-layer geometry.
// Usage: node scripts/verify-mlp-initialization.mjs <binary> <output>
import assert from 'node:assert/strict';
import {execFileSync, spawnSync} from 'node:child_process';
import {readFileSync, writeFileSync, existsSync, mkdirSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {createHash} from 'node:crypto';
const [binaryText, outputText] = process.argv.slice(2);
assert.ok(outputText);
const binary = resolve(binaryText), output = resolve(outputText);
assert.ok(!existsSync(output), 'Do not overwrite verification evidence.');
mkdirSync(output, {recursive: true});
const width = 16, layers = [[2, width], [width, width], [width, 1]], records = [];
for (const operation of ['div', 'mul']) for (const scale of ['raw', 'unit', 'centered']) {
  const initializations = new Map();
  for (const mode of ['zero', 'uniform', 'domain']) {
    const directory = join(output, `${operation}-${scale}-${mode}`);
    const args = ['--width', String(width), '--depth', '2', '--operation', operation, '--scale', scale,
      '--bias-init', mode, '--epochs', '2', '--report-every', '2', '--seed', '11', '--output', directory];
    const stdout = execFileSync(binary, args, {encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']});
    writeFileSync(join(directory, 'stdout.log'), stdout);
    writeFileSync(join(directory, 'verification-arguments.json'), JSON.stringify({binary, args}, null, 2) + '\n');
    const metadata = Object.fromEntries(readFileSync(join(directory, 'run.txt'), 'utf8').trim().split('\n').map(line => {
      const at = line.indexOf('='); return [line.slice(0, at), line.slice(at + 1)];
    }));
    assert.equal(metadata.status, 'complete');
    assert.equal(metadata.bias_initialization, mode);
    const initial = readFileSync(join(directory, 'initial-model.bin'));
    initializations.set(mode, initial);
    if (mode === 'domain') {
      const divisor = scale === 'raw' ? 1 : 96, shift = scale === 'centered' ? .5 : 0;
      const minA = -shift, minB = (operation === 'div' ? 1 : 0) / divisor - shift, maximum = 96 / divisor - shift;
      for (let channel = 0; channel < width; channel++) {
        const a = initial.readFloatLE(4 * channel), b = initial.readFloatLE(4 * (width + channel)), bias = initial.readFloatLE(4 * (2 * width + channel));
        const corners = [a * minA + b * minB + bias, a * minA + b * maximum + bias,
          a * maximum + b * minB + bias, a * maximum + b * maximum + bias];
        assert.ok(Math.min(...corners) < 0 && Math.max(...corners) > 0, 'First-layer hinge missed the operand box.');
      }
    }
  }
  let offset = 0;
  for (const [index, [inputs, outputs]] of layers.entries()) {
    const end = offset + inputs * outputs * 4;
    for (const weights of initializations.values()) assert.deepEqual(weights.subarray(offset, end), initializations.get('zero').subarray(offset, end), 'Kaiming matrix changed.');
    if (index > 0) assert.deepEqual(initializations.get('uniform').subarray(end, end + outputs * 4), initializations.get('domain').subarray(end, end + outputs * 4), 'Later biases changed.');
    offset = end + outputs * 4;
  }
  records.push({operation, scale, pairedMatrices: true, pairedLaterBiases: true, firstLayerDomainGeometry: true});
}
const invalid = spawnSync(binary, ['--bias-scale', '-1']);
assert.notEqual(invalid.status, 0);
assert.match(invalid.stderr.toString(), /bias scale/);
const sha = path => createHash('sha256').update(readFileSync(path)).digest('hex');
writeFileSync(join(output, 'summary.json'), JSON.stringify({binarySha256: sha(binary), scriptSha256: sha(new URL(import.meta.url)), nativeTrials: 18, records,
  invalidBiasScaleRejected: true, passed: true}, null, 2) + '\n');
console.log('PASS: 18 native initialization trials, identical Kaiming matrices and later biases, operand-box hinges, and invalid-scale rejection.');
