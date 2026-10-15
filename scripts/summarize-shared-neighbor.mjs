// Summarize checked Rust diagnostics and native predictions for the matched controls.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const destination = 'docs/grokking-results/shared-neighbor';
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const mean = values => {
  assert(values.length > 0);
  return values.reduce((a, b) => a + b, 0) / values.length;
};
const csv = path => {
  const [header, ...lines] = readFileSync(path, 'utf8').trim().split(/\r?\n/);
  const keys = header.split(',');
  return lines.map(line => {
    const values = line.split(',');
    assert.equal(values.length, keys.length);
    return Object.fromEntries(values.map((value, i) => {
      const number = Number(value);
      return [keys[i], value === '' || Number.isNaN(number) ? value : number];
    }));
  });
};

const runs = [];
for (const name of ['pre-norm', 'pre-norm-kaiming', 'pre-norm-no-decay', 'core-mixing']) {
  const root = join(destination, name);
  if (!existsSync(join(root, 'summary.json'))) continue;
  const archived = JSON.parse(readFileSync(join(root, 'summary.json'), 'utf8'));
  assert.equal(archived.runs.length, 1);
  const run = archived.runs[0];
  assert.equal(run.settings.status, 'complete');
  assert.equal(run.settings.normalization, 'pre_rmsnorm_blocks_and_final');
  assert.equal(run.settings.architecture, 'shared-neighbor-mesh-mlp');
  assert.equal(hash(join(run.directory, 'initial-model.bin')), run.initial_model_sha256);
  assert.equal(hash(join(run.directory, 'model.bin')), run.final_model_sha256);
  const verification = readFileSync(join(root, 'diagnostics/verification.txt'), 'utf8');
  for (const phase of ['initial', 'trained']) {
    const error = verification.match(new RegExp(`${phase}_reference_logit_maximum_absolute_error=(\\S+)`));
    assert(error && Number(error[1]) <= 1e-9);
  }
  const predictionPath = join(root, 'raw', run.trial, 'predictions.csv');
  copyFileSync(join(run.directory, 'predictions.csv'), predictionPath);
  const predictions = csv(predictionPath);
  assert.equal(predictions.length, 9312);
  const embeddings = csv(join(root, 'diagnostics/embedding-spread.csv'));
  const traces = csv(join(root, 'diagnostics/activation-traces.csv'));
  const gradients = csv(join(root, 'diagnostics/parameter-gradients.csv'));
  const phases = ['initial', 'trained'].map(phase => {
    const e = embeddings.filter(row => row.phase === phase);
    const g = gradients.filter(row => row.phase === phase && row.group === 'core');
    assert.equal(e.length, 97);
    assert.equal(g.length, 64);
    return {
      phase,
      embedding_norm_zero: e.find(row => row.residue === 0).l2,
      minimum_embedding_norm: Math.min(...e.map(row => row.l2)),
      maximum_embedding_norm: Math.max(...e.map(row => row.l2)),
      mean_effective_embedding_channels: mean(e.map(row => row.effective_channels_energy_participation)),
      selected_traces: traces.filter(row => row.phase === phase && [
        'joined', 'core_64', 'final_rmsnorm', 'centered_logits', 'answer_probabilities',
        'nonzero_a_centered_logits', 'nonzero_a_answer_probabilities',
      ].includes(row.stage)),
      minimum_core_gradient_rms: Math.min(...g.map(row => row.gradient_rms)),
      maximum_core_gradient_rms: Math.max(...g.map(row => row.gradient_rms)),
    };
  });
  const splits = ['train', 'validation'].map(split => {
    const rows = predictions.filter(row => row.split === split);
    assert.equal(rows.length, 4656);
    return {
      split, rows: rows.length,
      groups: [true, false].map(zero => {
        const group = rows.filter(row => (row.a === 0) === zero);
        return { a_zero: zero, rows: group.length, accuracy: mean(group.map(row => row.correct)) };
      }),
    };
  });
  const interactionPath = join(root, 'diagnostics/joint-interaction.csv');
  const interactions = existsSync(interactionPath) ? csv(interactionPath) : null;
  if (interactions) {
    assert.equal(interactions.length, 2);
    const actual = predictions.filter(row => row.a !== 0);
    assert.equal(actual.length, 9216);
    assert(Math.abs(interactions.find(row => row.phase === 'trained').nonzero_grid_accuracy
      - mean(actual.map(row => row.correct))) <= 1e-12);
  }
  runs.push({
    name, phases, splits, interactions,
    core_connectivity: run.settings.connectivity,
    prediction_scope: 'All 9312 equations, native CUDA predictions after 1000 epochs.',
    activation_scope: 'First 512 seeded training equations; gradients use first 16.',
    prediction_file: predictionPath,
    prediction_sha256: hash(predictionPath),
    initial_model_sha256: run.initial_model_sha256,
    final_model_sha256: run.final_model_sha256,
    trainer_sha256: archived.trainer_sha256,
    distinct_prediction_classes: new Set(predictions.map(row => row.prediction)).size,
  });
}
assert(runs.length > 0);
const report = {
  analysis_script_sha256: hash(fileURLToPath(import.meta.url)),
  diagnostic_source_sha256: hash('src/bin/shared-neighbor-diagnostics.rs'),
  interaction_source_sha256: hash('src/bin/shared-neighbor-interactions.rs'),
  runs,
};
const path = join(destination, 'diagnostic-summary.json');
writeFileSync(path, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ report: path, runs: runs.map(run => run.name) }));
