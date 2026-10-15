// Audit matched embedding initialization pairs and their measured learning results.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = 'docs/grokking-results/shared-neighbor/embedding-mixing';
const archive = JSON.parse(readFileSync(join(root, 'summary.json'), 'utf8'));
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const csv = path => {
  const [head, ...lines] = readFileSync(path, 'utf8').trim().split(/\r?\n/);
  const keys = head.split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((value, i) => [keys[i], Number.isNaN(Number(value)) ? value : Number(value)])));
};
const initial = csv(join(root, 'initialization/initialization-summary.csv'));
const keys = ['architecture', 'activation', 'encoding', 'operation', 'modulus', 'train_fraction', 'seed', 'init_seed', 'width', 'depth', 'embedding_mesh_depth', 'connectivity', 'mesh_initialization', 'parameters', 'normalization', 'core_structure', 'rmsnorm_epsilon', 'final_rmsnorm', 'learning_rate', 'weight_decay', 'betas', 'epsilon', 'warmup_steps', 'batch', 'requested_epochs', 'report_every', 'embedding_column_norm_target'];
const pairs = [];
for (const seed of [11, 17, 23]) {
  const find = mode => archive.runs.filter(run => Number(run.settings.seed) === seed && run.settings.shared_embedding_initialization === mode);
  const weak = find('unit-norm-weak'), strong = find('unit-norm-strong');
  assert(weak.length <= 1 && strong.length <= 1);
  if (weak.length !== 1 || strong.length !== 1) continue;
  const [a, b] = [weak[0], strong[0]];
  for (const key of keys) assert.equal(a.settings[key], b.settings[key], `${seed}: ${key}`);
  const pa = readFileSync(join(a.directory, 'initial-model.bin'));
  const pb = readFileSync(join(b.directory, 'initial-model.bin'));
  const offset = 4 * 2 * Number(a.settings.width) * Number(a.settings.embedding_mesh_depth);
  assert(pa.subarray(offset).equals(pb.subarray(offset)));
  for (const run of [a,b]) {
    const mode = run.settings.shared_embedding_initialization;
    const expected = join(root, 'initialization', `${mode}-seed${seed}`, 'initial-model.bin');
    assert.equal(hash(expected), run.initial_model_sha256);
    assert.equal(hash(join(run.directory, 'initial-model.bin')), run.initial_model_sha256);
    const geometry = initial.find(row => row.mode === mode && row.seed === seed);
    assert(geometry && Math.abs(geometry.min_norm - 1) < 1e-6 && Math.abs(geometry.max_norm - 1) < 1e-6);
    assert(Math.abs(geometry.effective_gram_rank - 97) < 1e-5);
  }
  const result = run => ({
    trial: run.trial, final: run.statistics.final,
    first_train_99: run.statistics.first_train_99,
    grokking_matched: run.statistics.matched,
    initial_geometry: initial.find(row => row.mode === run.settings.shared_embedding_initialization && row.seed === seed),
    initial_model_sha256: run.initial_model_sha256,
    final_model_sha256: run.final_model_sha256,
    history_sha256: run.history_sha256,
  });
  pairs.push({seed, checked_settings: keys, core_parameters_identical: true, cpu_gpu_initial_models_identical: true,
    weak: result(a), strong: result(b),
    train_accuracy_delta: b.statistics.final.train - a.statistics.final.train,
    test_accuracy_delta: b.statistics.final.test - a.statistics.final.test});
}
const mean = key => pairs.length ? pairs.reduce((sum, pair) => sum + pair[key], 0) / pairs.length : null;
const report = {
  scope: 'Paired 1000-epoch pilots; core, split, sampler and optimizer fixed within each seed. Initial class norms and Gram energy rank matched. Orthogonality is initialization-only.',
  expected_pairs: 3, completed_pairs: pairs.length,
  trainer_sha256: archive.trainer_sha256,
  analysis_script_sha256: hash(fileURLToPath(import.meta.url)),
  mean_train_accuracy_delta: mean('train_accuracy_delta'),
  mean_test_accuracy_delta: mean('test_accuracy_delta'), pairs,
};
writeFileSync(join(root, 'paired-summary.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({completed_pairs: pairs.length, pairs: pairs.map(pair => ({seed:pair.seed, weak_train:pair.weak.final.train, strong_train:pair.strong.final.train, weak_test:pair.weak.final.test, strong_test:pair.strong.final.test}))}, null, 2));
