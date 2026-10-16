// Re-evaluate immutable CUDA checkpoints and independently audit both data splits.
// Usage: node scripts/verify-mlp-checkpoints.mjs EVALUATOR NEW_OUTPUT RUN_DIRECTORY...
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFileSync, writeFileSync, existsSync, mkdirSync} from 'node:fs';
import {resolve, join, basename} from 'node:path';
import {createHash} from 'node:crypto';

const [binaryText, outputText, ...directories] = process.argv.slice(2);
assert.ok(binaryText && outputText && directories.length, 'Provide an evaluator, a new output directory and completed runs.');
const binary = resolve(binaryText), output = resolve(outputText);
assert.ok(!existsSync(output), 'Do not overwrite earlier checkpoint verification.');
mkdirSync(output, {recursive: true});
const sha = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const csv = path => {
  const [header, ...lines] = readFileSync(path, 'utf8').trim().split('\n'), fields = header.split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [fields[index], Number(value)])));
};
const round = value => Math.trunc(value + (value >= 0 ? .5 : -.5));
const near = (actual, expected, tolerance, label) => {
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected));
  assert.ok(Math.abs(actual - expected) <= tolerance, `${label}: ${actual} versus ${expected}`);
};
const records = [], evaluatorSha256 = sha(binary);
for (const text of directories) {
  const directory = resolve(text), evaluationDirectory = join(output, basename(directory));
  const manifest = Object.fromEntries(readFileSync(join(directory, 'run.txt'), 'utf8').trim().split('\n').map(line => {
    const at = line.indexOf('='); return [line.slice(0, at), line.slice(at + 1)];
  }));
  assert.equal(manifest.status, 'complete');
  const files = ['model.bin', 'initial-model.bin', 'loss.csv', 'run.txt', 'predictions.csv', 'arguments.json'];
  const inputSha256 = Object.fromEntries(files.map(file => [file, sha(join(directory, file))]));
  const args = [directory, evaluationDirectory];
  const stdout = execFileSync(binary, args, {encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']});
  writeFileSync(join(evaluationDirectory, 'stdout.json'), stdout);
  const native = JSON.parse(stdout), final = csv(join(directory, 'loss.csv')).at(-1);
  assert.equal(native.operation, manifest.operation);
  for (const field of ['modulus', 'width', 'depth', 'seed', 'parameters']) assert.equal(native[field], Number(manifest[field]));
  assert.equal(final.epoch, Number(manifest.requested_epochs));
  assert.equal(native.unusedTrainingRows, Number(manifest.unused_training_equations ?? 0));
  const divisor = manifest.scale === 'raw' ? 1 : Number(manifest.modulus) - 1;
  const offset = manifest.scale === 'centered' ? .5 : 0, pairs = new Set(), splitMetrics = {}, predictionSha256 = {};
  for (const split of ['train', 'test']) {
    const file = `${split}-predictions.csv`, rows = csv(join(evaluationDirectory, file));
    predictionSha256[file] = sha(join(evaluationDirectory, file));
    assert.equal(rows.length, Number(manifest[split === 'train' ? 'training_equations' : 'validation_equations']));
    const original = split === 'test' ? csv(join(directory, 'predictions.csv')) : null;
    let squares = 0, mse = 0, correct = 0;
    for (const [index, row] of rows.entries()) {
      for (const field of ['a', 'b', 'target']) assert.ok(Number.isInteger(row[field]) && row[field] >= 0 && row[field] < native.modulus);
      assert.ok(Number.isFinite(row.prediction_residue));
      const pair = `${row.a}/${row.b}`;
      assert.ok(!pairs.has(pair), 'Repeated equation or train/held-out overlap.'); pairs.add(pair);
      if (native.operation === 'div') assert.ok(row.b > 0 && row.target * row.b % native.modulus === row.a);
      else {assert.equal(native.operation, 'mul'); assert.equal(row.target, row.a * row.b % native.modulus);}
      squares += (row.prediction_residue - row.target) ** 2;
      const prediction = row.prediction_residue / divisor - offset;
      const target = Math.fround(Math.fround(row.target / divisor) - offset);
      mse += (prediction - target) ** 2;
      correct += Number(round(row.prediction_residue) === row.target);
      if (original) {
        for (const field of ['a', 'b', 'target']) assert.equal(row[field], original[index][field]);
        near(row.prediction_residue, original[index].prediction_residue, 6e-10, 'Saved held-out prediction');
      }
    }
    const metrics = {rows: rows.length, mse: mse / rows.length, rmse: Math.sqrt(squares / rows.length), accuracy: correct / rows.length};
    assert.equal(native[split].rows, rows.length);
    for (const field of ['mse', 'rmse', 'accuracy']) {
      near(metrics[field], native[split][field], 1e-10 * Math.max(1, metrics[field]), `${split} exported ${field}`);
      const recordedField = `${split}_${field === 'rmse' ? 'rmse_residue' : field}`;
      near(native[split][field], final[recordedField], field === 'mse' ? 6e-13 : 6e-10, `${split} checkpoint versus training log ${field}`);
    }
    splitMetrics[split] = metrics;
  }
  assert.equal(pairs.size + native.unusedTrainingRows, native.modulus * (native.modulus - Number(native.operation === 'div')));
  for (const file of files) assert.equal(sha(join(directory, file)), inputSha256[file], 'Original training evidence changed.');
  records.push({directory, evaluationDirectory, inputSha256, predictionSha256, stdoutSha256: sha(join(evaluationDirectory, 'stdout.json')), args, native, independentlyRecomputed: splitMetrics, passed: true});
  console.log(`PASS ${native.operation} W${native.width} D${native.depth} seed ${native.seed}: train ${(100 * native.train.accuracy).toFixed(2)}%, held-out ${(100 * native.test.accuracy).toFixed(2)}%.`);
}
assert.equal(sha(binary), evaluatorSha256);
writeFileSync(join(output, 'summary.json'), JSON.stringify({binary, evaluatorSha256, records, passed: true}, null, 2) + '\n');
