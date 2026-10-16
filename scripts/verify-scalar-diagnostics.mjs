// Verify default replay and diagnostic data/scaling paths with native CUDA runs.
// Usage: node scripts/verify-scalar-diagnostics.mjs <original> <diagnostic> <output>
import assert from 'node:assert/strict';
import {execFileSync, spawnSync} from 'node:child_process';
import {readFileSync, writeFileSync, mkdirSync, existsSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {createHash} from 'node:crypto';

const [originalText, diagnosticText, outputText] = process.argv.slice(2);
assert.ok(outputText, 'Provide both binaries and a new local output directory.');
const original = resolve(originalText), diagnostic = resolve(diagnosticText), output = resolve(outputText);
assert.ok(!existsSync(output), 'Do not overwrite verification evidence.');
mkdirSync(output, {recursive: true});
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const rows = directory => {
  const [header, ...lines] = readFileSync(join(directory, 'loss.csv'), 'utf8').trim().split('\n');
  return lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [header.split(',')[index], Number(value)])));
};
const metadata = directory => Object.fromEntries(readFileSync(join(directory, 'run.txt'), 'utf8').trim().split('\n').map(line => {
  const at = line.indexOf('='); return [line.slice(0, at), line.slice(at + 1)];
}));
const run = (binary, name, arguments_) => {
  const directory = join(output, name);
  const args = [...arguments_, '--output', directory];
  const stdout = execFileSync(binary, args, {encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']});
  writeFileSync(join(directory, 'verification-arguments.json'), JSON.stringify({binary, args}, null, 2) + '\n');
  writeFileSync(join(directory, 'stdout.log'), stdout);
  assert.equal(metadata(directory).status, 'complete');
  return directory;
};
const defaults = [];
for (const operation of ['div', 'mul']) for (const scale of ['unit', 'raw']) {
  const args = ['--width', '16', '--depth', '2', '--operation', operation, '--scale', scale,
    '--epochs', '10', '--batch', '512', '--report-every', '10', '--seed', '11'];
  const earlier = run(original, `original-${operation}-${scale}`, args);
  const current = run(diagnostic, `diagnostic-${operation}-${scale}`, args);
  const strip = directory => rows(directory).map(({elapsed_seconds, ...row}) => row);
  assert.deepEqual(strip(current), strip(earlier), 'Default metrics changed.');
  for (const file of ['initial-model.bin', 'model.bin', 'predictions.csv'])
    assert.equal(hash(join(current, file)), hash(join(earlier, file)), `Default ${file} changed.`);
  defaults.push({operation, scale, metricsAndCheckpointsIdentical: true});
}
const controls = [];
for (const operation of ['div', 'mul']) for (const scale of ['raw', 'unit', 'centered']) {
  const directory = run(diagnostic, `subset-${operation}-${scale}`, ['--width', '16', '--depth', '2',
    '--operation', operation, '--modulus', '7', '--train-limit', '8', '--scale', scale,
    '--epochs', '3', '--batch', '5', '--report-every', '2', '--schedule', 'cosine', '--beta2', '.999', '--seed', '11']);
  const recorded = rows(directory), manifest = metadata(directory);
  assert.deepEqual(recorded.map(row => [row.epoch, row.step]), [[0, 0], [2, 4], [3, 6]]);
  assert.equal(manifest.training_equations, '8');
  assert.equal(Number(manifest.validation_equations), operation === 'div' ? 21 : 25);
  assert.equal(manifest.schedule, 'cosine');
  assert.equal(manifest.betas, '0.9,0.999');
  const predictions = readFileSync(join(directory, 'predictions.csv'), 'utf8').trim().split('\n').slice(1)
    .map(line => line.split(',').map(Number));
  let squares = 0, correct = 0;
  for (const [a, b, target, prediction] of predictions) {
    const expected = operation === 'mul' ? a * b % 7 : Array.from({length: 7}, (_, answer) => answer).find(answer => answer * b % 7 === a);
    assert.equal(target, expected, 'Incorrect modular target.');
    squares += (prediction - target) ** 2;
    correct += Number(Math.trunc(prediction + (prediction >= 0 ? .5 : -.5)) === target);
  }
  const mse = squares / predictions.length, divisor = scale === 'raw' ? 1 : 6;
  assert.ok(Math.abs(Math.sqrt(mse) - recorded.at(-1).test_rmse_residue) < 1e-6);
  assert.ok(Math.abs(mse - recorded.at(-1).test_mse * divisor ** 2) < 1e-5 * Math.max(1, mse));
  assert.ok(Math.abs(correct / predictions.length - recorded.at(-1).test_accuracy) < 1e-8);
  controls.push({operation, scale, targetsScalingTailAndPartialReportVerified: true});
}
const rejection = spawnSync(diagnostic, ['--operation', 'div', '--modulus', '7', '--train-limit', '100']);
assert.notEqual(rejection.status, 0);
assert.match(rejection.stderr.toString(), /train limit/);
const summary = {originalSha256: hash(original), diagnosticSha256: hash(diagnostic), defaults, controls,
  invalidSubsetRejected: true, passed: true};
writeFileSync(join(output, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
console.log('PASS: four exact default replays, six data/scaling/tail controls, and invalid-subset rejection.');
