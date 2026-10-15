// Compare the width-normal E control with the archived uniform-E Rust runs.
// The independent browser f64 forward pass reads native checkpoints; no web
// initialization or frontend training behavior is changed by this analysis.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {existsSync, readFileSync, readdirSync, writeFileSync, mkdirSync} from 'node:fs';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {TRAIN, dataset} from '../web/own-channel/dist/model.js';
function heldoutProbes(seed) {
  const table = dataset(seed), rows = [];
  for (let i = TRAIN; i < TRAIN + 128; i++) rows.push(Array.from(table.subarray(i * 3, i * 3 + 3)));
  rows.sort((a, b) => a[2] - b[2] || a[0] - b[0] || a[1] - b[1]);
  return Uint32Array.from(rows.flat());
}
import {predictModel} from '../web/own-channel/dist/models.js';

const initialOnly = process.argv.includes('--initial-only');
const study = 'target/grok-div97-dense-embedding-normal-width';
const destination = 'docs/grokking-results/dense-embedding-normal-width';
const reference = JSON.parse(readFileSync('docs/grokking-results/rotary-dense-relu/summary.json', 'utf8'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const metadata = directory => Object.fromEntries(readFileSync(join(directory, 'run.txt'), 'utf8').trim().split('\n').filter(line => line.includes('=')).map(line => {
  const at = line.indexOf('='); return [line.slice(0, at), line.slice(at + 1)];
}));
const settingsKeys = ['architecture', 'activation', 'encoding', 'operation', 'modulus', 'train_fraction', 'seed', 'init_seed', 'width', 'depth', 'parameters', 'all_parameters', 'normalization', 'bias', 'learning_rate', 'weight_decay', 'betas', 'epsilon', 'warmup_steps', 'batch', 'requested_epochs', 'report_every', 'steps_per_epoch', 'stop_validation_accuracy'];
const geometry = (weights, width, seed) => {
  const E = weights.subarray(0, 97 * width), mean = E.reduce((sum, value) => sum + value, 0) / E.length;
  const variance = E.reduce((sum, value) => sum + (value - mean) ** 2, 0) / E.length;
  const norms = Array.from({length: 97}, (_, k) => Math.sqrt(E.subarray(k * width, (k + 1) * width).reduce((sum, value) => sum + value * value, 0)));
  const pairs = heldoutProbes(seed), model = width === 128 ? 'dense-rotary' : 'dense-rotary-wide';
  let logitVariance = 0, loss = 0, inputRms = 0;
  for (let i = 0; i < pairs.length; i += 3) {
    const prediction = predictModel(weights, pairs[i], pairs[i + 1], model);
    const average = prediction.logits.reduce((sum, value) => sum + value, 0) / 97, maximum = Math.max(...prediction.logits);
    logitVariance += prediction.logits.reduce((sum, value) => sum + (value - average) ** 2, 0) / 97;
    loss += Math.log(prediction.logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0)) + maximum - prediction.logits[pairs[i + 2]];
    inputRms += Math.sqrt(prediction.rawInput.reduce((sum, value) => sum + value * value, 0) / width);
  }
  return {embeddingMean: mean, embeddingVariance: variance, embeddingStd: Math.sqrt(variance),
    minClassNorm: Math.min(...norms), meanClassNorm: norms.reduce((sum, value) => sum + value, 0) / 97, maxClassNorm: Math.max(...norms),
    diagnosticSampleCount: pairs.length / 3, diagnosticSplit: '128 fixed held-out pairs',
    meanInputRms: inputRms / (pairs.length / 3), centeredLogitRms: Math.sqrt(logitVariance / (pairs.length / 3)),
    crossEntropy: loss / (pairs.length / 3), precision: 'independent f64 inference on native f32 weights'};
};

const pairs = [];
const archive = !initialOnly ? JSON.parse(readFileSync(join(destination, 'summary.json'), 'utf8')) : null;
for (const width of [128, 1024]) {
  const baseline = reference.runs.find(run => Number(run.settings.width) === width && Number(run.settings.seed) === 11);
  assert(baseline);
  const directories = readdirSync(join(study, 'trials')).map(name => join(study, 'trials', name)).filter(directory => existsSync(join(directory, 'initial-model.bin')) && existsSync(join(directory, 'run.txt')));
  const candidates = directories.map(directory => ({directory, settings: metadata(directory)})).filter(run => Number(run.settings.width) === width && Number(run.settings.seed) === 11);
  if (initialOnly && !candidates.length) continue;
  assert.equal(candidates.length, 1, `normal-width W${width} run`);
  const run = candidates[0];
  assert.equal(run.settings.dense_embedding_initialization, 'normal-width');
  for (const key of settingsKeys) assert.equal(baseline.settings[key], run.settings[key], `W${width}: ${key}`);
  const a = readFileSync(join(baseline.directory, 'initial-model.bin')), b = readFileSync(join(run.directory, 'initial-model.bin'));
  assert.equal(a.length, 4 * Number(run.settings.parameters)); assert.equal(b.length, a.length);
  assert(a.subarray(4 * 97 * width).equals(b.subarray(4 * 97 * width)), `W${width} dense A changed`);
  const baselineHistory = readFileSync(join(baseline.directory, 'loss.csv'));
  assert.equal(hash(baselineHistory), baseline.history_sha256);
  const baselineInitialHash = hash(a);
  assert.equal(baselineInitialHash, baseline.initial_model_sha256);
  const datasetHash = hash(readFileSync(join(baseline.directory, 'equations.csv')));
  assert.equal(datasetHash, hash(readFileSync(join(run.directory, 'equations.csv'))));
  const asFloats = bytes => new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const oldGeometry = geometry(asFloats(a), width, 11), newGeometry = geometry(asFloats(b), width, 11);
  assert(Math.abs(newGeometry.embeddingVariance * width - 1) < .025);
  const result = {width, seed: 11, checkedSettings: settingsKeys, identicalDenseA: true,
    datasetSha256: datasetHash, denseASha256: hash(a.subarray(4 * 97 * width)),
    uniform: {initialSha256: baselineInitialHash, geometry: oldGeometry, statistics: baseline.statistics},
    normal: {initialSha256: hash(b), geometry: newGeometry}};
  if (!initialOnly) {
    const archived = archive.runs.find(item => item.directory === run.directory || item.directory.endsWith('/' + run.directory));
    assert(archived && run.settings.status === 'complete');
    assert.equal(hash(b), archived.initial_model_sha256);
    assert.equal(hash(readFileSync(join(run.directory, 'loss.csv'))), archived.history_sha256);
    assert.equal(hash(readFileSync(join(run.directory, 'model.bin'))), archived.final_model_sha256);
    result.normal.statistics = archived.statistics;
    result.normal.historySha256 = archived.history_sha256;
    result.normal.finalSha256 = archived.final_model_sha256;
  }
  pairs.push(result);
}
mkdirSync(destination, {recursive: true});
const report = {scope: 'Single-seed Rust/CUDA E-distribution and scale control at both widths. Initial A, data split, sampler and optimizer fixed. Variance is an initialization property; no norm constraint during training. Both distribution shape and variance change.',
  initialOnly, completedPairs: pairs.length, analysisScriptSha256: hash(readFileSync(fileURLToPath(import.meta.url))),
  trainerSha256: hash(readFileSync(join(study, 'bin/mesh-gpt'))),
  independentForwardSourceSha256: hash(readFileSync('web/own-channel/dist/models.js')), pairs};
writeFileSync(join(destination, initialOnly ? 'initial-comparison.json' : 'paired-comparison.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(pairs.map(pair => ({width:pair.width, denseAIdentical:pair.identicalDenseA,
  oldLogitRms:pair.uniform.geometry.centeredLogitRms, newLogitRms:pair.normal.geometry.centeredLogitRms,
  oldInitialLoss:pair.uniform.geometry.crossEntropy, newInitialLoss:pair.normal.geometry.crossEntropy,
  firstTest99:pair.normal.statistics?.first_test_99?.epoch, matched:pair.normal.statistics?.matched})), null, 2));

if (!initialOnly) {
  assert.equal(pairs.length, 2);
  const epoch = row => row?.epoch ?? 'None';
  const interval = stats => stats.plateau.start === null ? 'None' : `${stats.plateau.start}–${stats.plateau.end}: ${stats.plateau.epochs} epochs`;
  const rows = pairs.flatMap(pair => [['Uniform', pair.uniform], ['Normal 1/W', pair.normal]].map(([label, run]) => {
    const s = run.statistics;
    return `| ${pair.width} | ${label} | ${epoch(s.first_train_99)} | ${interval(s)} | ${epoch(s.first_test_90)} | ${s.rise_epochs ?? 'None'} | ${epoch(s.first_test_99)} | ${s.matched ? 'Yes' : 'No'} |`;
  }));
  const initialRows = pairs.flatMap(pair => [['Uniform', pair.uniform], ['Normal 1/W', pair.normal]].map(([label, run]) => {
    const g = run.geometry;
    return `| ${pair.width} | ${label} | ${g.embeddingStd.toFixed(5)} | ${g.meanClassNorm.toFixed(3)} | ${g.centeredLogitRms.toFixed(3)} | ${g.crossEntropy.toFixed(3)} |`;
  }));
  const content = [
    '# Dense + ReLU: width-normal shared embedding control', '',
    'Two completed Rust/CUDA runs replace the shared E initializer with Gaussian',
    'variance 1/W, at widths 128 and 1024. Both use seed 11 and are compared with',
    'the archived uniform-E runs of the same Dense + ReLU model. The initializer',
    'is now used by both local frontend tabs. Rust retains its opt-in switch.', '',
    'The unchanged model is x0 = E[a] + J(E[b]), r = RMSNorm(x0),',
    'x1 = x0 + ReLU(r A), logits = RMSNorm(x1) E^T. There are no biases or learned',
    'normalization gains. Pre-RMSNorm, the raw skip and final RMSNorm remain.', '',
    '## Matched control', '',
    '- Baseline E: uniform in [-1/sqrt(97), +1/sqrt(97)], variance 1/(3*97).',
    '- New E: Normal(0, variance=1/W), standard deviation 1/sqrt(W).',
    '- A: byte-identical Kaiming-normal matrix with variance 2/W in each pair.',
    '- Data/sampling seed, split, architecture, parameter count and optimizer match.',
    '- No embedding norm constraint is imposed during training.', '',
    'Box-Muller draws consume the same number of uniforms as the old E',
    'initializer, preserving the exact A random stream. The paired audit verifies',
    'A bytes, dataset hashes, initial/final checkpoint hashes and history hashes.',
    'The control changes distribution shape and variance together. Scaling E also',
    'changes the raw skip relative to the normalized residual branch; it is not',
    'a readout-temperature-only intervention.', '',
    'Both runs use division modulo 97, 4,656 training and 4,656 held-out pairs,',
    'batch 512 with a 48-row tail, AdamW learning rate 0.001, weight decay 0.1,',
    'betas 0.9/0.98, epsilon 1e-8 and ten warmup updates. Both complete splits are',
    'evaluated every epoch. The cap is 1,000 epochs, stopping at test accuracy >=99%.', '',
    '## Initial geometry', '',
    '| Width | E initializer | E std | Mean class norm | Centered logit RMS | Probe CE |',
    '| --- | --- | ---: | ---: | ---: | ---: |', ...initialRows, '',
    'Logit RMS and cross entropy use independent f64 inference on native f32',
    'checkpoints, for the same 128 fixed held-out pairs. The uniform predictor',
    'has CE log(97) = 4.575. E column/row norms are sampled Gaussian quantities,',
    'not fixed unit norms. Normal 1/W makes the initial logit scale close to one',
    'at both widths; that observation alone does not establish better learning.', '',
    '## Complete training results', '',
    '| Width | E initializer | First train >=99% | Train >=99%, test <=10% plateau | First test >=90% | Rise epochs | First test >=99% | Grokking |',
    '| --- | --- | ---: | --- | ---: | ---: | ---: | --- |', ...rows, '',
    'The unchanged criterion requires at least 25 elapsed continuous plateau',
    'epochs, a rise from the last test <=10% to the first test >=90% within',
    '250 epochs, and a later test >=99%. Missing observations or train dips',
    'break the plateau. These results cover one seed per width and do not',
    'establish seed robustness or browser crossing epochs.', '',
    '![Old and new shared E learning curves](grokking-results/dense-embedding-normal-width/dense-embedding-comparison.png)', '',
    '[SVG comparison](grokking-results/dense-embedding-normal-width/dense-embedding-comparison.svg),',
    '[paired initialization and result audit](grokking-results/dense-embedding-normal-width/paired-comparison.json),',
    '[full histories and provenance](grokking-results/dense-embedding-normal-width/summary.json),',
    'and [baseline results](rotary-dense-results.md) preserve the measured outcomes.', '',
    '## Reproduce and validate', '',
    'The queue is [div97-dense-embedding-normal-width.args](../experiments/div97-dense-embedding-normal-width.args).',
    'The new Rust option is `--dense-embedding-initialization normal-width`;',
    'the explicit baseline option is `uniform-fan-in`.', '',
    'Validation passed 91 Rust tests, strict all-target Clippy, formatting and',
    'the locked sm_86 CUDA build. Four Dense GPU/f64 verification fixtures check',
    'logits, loss, every gradient, gathered/offset minibatches, AdamW and reset',
    'repeatability, including normal-width E at the actual W128/modulus97 shape.',
    '[Validation logs](grokking-results/dense-embedding-normal-width/validation/)',
    'and [source hashes](grokking-results/dense-embedding-normal-width/source.json) are archived.', '',
  ].join('\n');
  writeFileSync('docs/rotary-dense-embedding-results.md', content);
}
