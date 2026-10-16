// Validate the local experiment evidence and write compact tables and a report.
// Usage: node scripts/summarize-local-ablations.mjs <output>
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync, existsSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {createHash} from 'node:crypto';

const base = resolve(process.argv[2] ?? 'target/local-ablations-2026-10-16');
const read = path => JSON.parse(readFileSync(join(base, path)));
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const csv = path => {const [header, ...lines] = readFileSync(path, 'utf8').trim().split('\n'), fields = header.split(','); return lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [fields[index], Number(value)])));};
assert.ok(read('rotary/source.json').completed && read('scalar/source.json').completed, 'Finish both experiments before summarizing.');
const rotary = read('rotary/summary.json'), scalar = read('scalar/summary.json');
const scalarRuns = read('scalar/runs.json');
const log = name => existsSync(join(base, name)) ? readFileSync(join(base, name), 'utf8') : '';
const verification = {
  rotaryCudaChecks: log('rotary-verification.log').split('\n').filter(line => line.startsWith('PASS ')).length,
  scalarCudaChecks: log('scalar-verification.log').split('\n').filter(line => line.startsWith('PASS ')).length,
  rustTests: Number(log('tests-final.log').match(/test result: ok\. (\d+) passed/)?.[1] ?? 0),
  strictClippy: log('clippy.log').includes('Finished') && !log('clippy.log').includes('error:'),
  exactRotaryReplay: existsSync(join(base, 'rotary-control-replay.json')) ? read('rotary-control-replay.json') : null,
  affineControl: existsSync(join(base, 'scalar-affine-control.json')) ? read('scalar-affine-control.json') : null,
};
const initializations = new Map();
const heldoutTargetGroups = new Map();
const firstLayerRows = [];
function firstLayerGeometry(weights, run) {
  const width = run.width, divisor = run.scale === 'raw' ? 1 : 96;
  const minA = 0, minB = run.operation === 'div' ? 1 / divisor : 0, maximum = 96 / divisor;
  const dead = [], linear = [], crossing = [];
  for (let channel = 0; channel < width; channel++) {
    const a = weights.readFloatLE(4 * channel), b = weights.readFloatLE(4 * (width + channel)), bias = weights.readFloatLE(4 * (2 * width + channel));
    const corners = [a * minA + b * minB + bias, a * minA + b * maximum + bias,
      a * maximum + b * minB + bias, a * maximum + b * maximum + bias];
    if (Math.max(...corners) <= 0) dead.push(channel);
    else if (Math.min(...corners) >= 0) linear.push(channel);
    else crossing.push(channel);
  }
  return {globallyDead: dead.length, globallyLinear: linear.length, signCrossing: crossing.length, deadChannels: dead};
}
const trainingFit = {screen: {bestFinal: null, bestObserved: null}, final: {bestFinal: null, bestObserved: null}};
for (const run of scalarRuns) {
  assert.equal(hash(join(run.directory, 'loss.csv')), run.historySha256, 'History changed since collection.');
  const weights = readFileSync(join(run.directory, 'initial-model.bin'));
  assert.equal(weights.length, run.parameters * 4);
  const initializationKey = `${run.width}/${run.depth}/${run.seed}`, initializationHash = hash(join(run.directory, 'initial-model.bin'));
  if (initializations.has(initializationKey)) assert.equal(initializationHash, initializations.get(initializationKey), 'Paired initialization changed.');
  initializations.set(initializationKey, initializationHash);
  if (run.status !== 'complete') continue;
  const fit = trainingFit[run.stage];
  if (!fit.bestFinal || run.finalTrainAccuracy > fit.bestFinal.accuracy)
    fit.bestFinal = {accuracy: run.finalTrainAccuracy, testAccuracy: run.finalTestAccuracy, directory: run.directory};
  for (const row of csv(join(run.directory, 'loss.csv')))
    if (!fit.bestObserved || row.train_accuracy > fit.bestObserved.accuracy)
      fit.bestObserved = {accuracy: row.train_accuracy, epoch: row.epoch, directory: run.directory};
  const trainedWeights = readFileSync(join(run.directory, 'model.bin'));
  assert.equal(trainedWeights.length, run.parameters * 4);
  const initialGeometry = firstLayerGeometry(weights, run), trainedGeometry = firstLayerGeometry(trainedWeights, run);
  assert.ok(initialGeometry.deadChannels.every(channel => trainedGeometry.deadChannels.includes(channel)), 'A globally inactive initial ReLU channel unexpectedly became active.');
  if (run.stage === 'final') firstLayerRows.push({directory: run.directory, operation: run.operation, scale: run.scale,
    width: run.width, depth: run.depth, seed: run.seed, initial: initialGeometry, trained: trainedGeometry});
  const predictions = csv(join(run.directory, 'predictions.csv'));
  assert.equal(predictions.length, run.operation === 'div' ? 4656 : 4705);
  const pairs = new Set();
  for (const row of predictions) {
    assert.ok([row.a, row.b, row.target].every(value => Number.isInteger(value) && value >= 0 && value < 97));
    assert.ok(Number.isFinite(row.prediction_residue));
    const pair = `${row.a}/${row.b}`;
    assert.ok(!pairs.has(pair), 'Duplicate held-out equation.');
    pairs.add(pair);
    if (run.operation === 'div') assert.ok(row.b > 0 && row.target * row.b % 97 === row.a, 'Incorrect division target.');
    else assert.equal(row.target, row.a * row.b % 97, 'Incorrect multiplication target.');
  }
  const groups = Object.fromEntries(['zero', 'nonzero'].map(name => {
    const own = predictions.filter(row => (row.target === 0) === (name === 'zero'));
    const correct = own.filter(row => Math.trunc(row.prediction_residue + (row.prediction_residue >= 0 ? .5 : -.5)) === row.target).length;
    return [name, {rows: own.length, accuracy: correct / own.length,
      rmse: Math.sqrt(own.reduce((sum, row) => sum + (row.prediction_residue - row.target) ** 2, 0) / own.length)}];
  }));
  heldoutTargetGroups.set(run.directory, groups);
  const rmse = Math.sqrt(predictions.reduce((sum, row) => sum + (row.prediction_residue - row.target) ** 2, 0) / predictions.length);
  const accuracy = predictions.filter(row => Math.trunc(row.prediction_residue + (row.prediction_residue >= 0 ? .5 : -.5)) === row.target).length / predictions.length;
  assert.ok(Math.abs(rmse - run.finalTestRmse) < 2e-6 * Math.max(1, rmse), 'Held-out RMSE differs from saved predictions.');
  assert.ok(Math.abs(accuracy - run.finalTestAccuracy) < 1e-8, 'Rounded accuracy differs from saved predictions.');
  if (run.stage === 'final' && run.seed === scalar.plan.scalarMlp.screenSeed) {
    const fields = ['operation', 'scale', 'width', 'depth', 'preRmsnorm', 'residual', 'learningRate', 'weightDecay'];
    const screen = scalarRuns.find(candidate => candidate.stage === 'screen' && fields.every(field => candidate[field] === run[field]));
    assert.ok(screen);
    const strip = path => readFileSync(path, 'utf8').trim().split('\n').map(line => line.split(',').slice(0, -1).join(','));
    const earlier = strip(join(screen.directory, 'loss.csv')), replay = strip(join(run.directory, 'loss.csv'));
    assert.deepEqual(replay.slice(0, earlier.length), earlier, 'Full run did not reproduce the screening prefix.');
  }
}
for (const run of rotary.runs) assert.equal(hash(join(run.directory, 'loss.csv')), run.historySha256);
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const rotaryRows = rotary.plan.profiles.map(profile => ({...profile,
  withResidual: rotary.seeds.map(seed => rotary.runs.find(run => run.profile === profile.name && run.seed === seed && run.residual)),
  withoutResidual: rotary.seeds.map(seed => rotary.runs.find(run => run.profile === profile.name && run.seed === seed && !run.residual)),
}));
for (const profile of rotaryRows) for (const [index, withSkip] of profile.withResidual.entries())
  assert.equal(hash(join(withSkip.directory, 'initial-model.bin')), hash(join(profile.withoutResidual[index].directory, 'initial-model.bin')), 'W128 paired initial weights differ.');
const families = scalar.results.map(result => ({family: result.family,
  configuration: Object.fromEntries(['width', 'depth', 'preRmsnorm', 'residual', 'learningRate', 'weightDecay', 'parameters'].map(field => [field, result.screen[field]])),
  seeds: result.runs.map(run => ({seed: run.seed, status: run.status, trainRmse: run.finalTrainRmse, testRmse: run.finalTestRmse,
    trainAccuracy: run.finalTrainAccuracy, testAccuracy: run.finalTestAccuracy, heldoutTargetGroups: heldoutTargetGroups.get(run.directory) ?? null})),
  meanTrainRmse: result.runs.every(run => run.status === 'complete') ? mean(result.runs.map(run => run.finalTrainRmse)) : null,
  meanTrainAccuracy: result.runs.every(run => run.status === 'complete') ? mean(result.runs.map(run => run.finalTrainAccuracy)) : null,
  meanTestRmse: result.runs.every(run => run.status === 'complete') ? mean(result.runs.map(run => run.finalTestRmse)) : null,
  meanTestAccuracy: result.runs.every(run => run.status === 'complete') ? mean(result.runs.map(run => run.finalTestAccuracy)) : null,
  meanConstantTestRmse: mean(result.runs.map(run => run.constantTestRmse)),
}));
const screenEffects = [];
for (const operation of scalar.plan.scalarMlp.operations) for (const scale of scalar.plan.scalarMlp.scales)
  for (const flag of ['preRmsnorm', 'residual']) {
    const fields = ['operation', 'scale', 'width', 'depth', 'preRmsnorm', 'residual', 'learningRate', 'weightDecay'].filter(field => field !== flag);
    const enabled = scalarRuns.filter(run => run.stage === 'screen' && run.operation === operation && run.scale === scale && run.depth > 1 && run[flag]);
    const differences = enabled.flatMap(on => {
      const off = scalarRuns.find(run => run.stage === 'screen' && !run[flag] && fields.every(field => run[field] === on[field]));
      assert.ok(off, 'Missing paired norm/skip screen.');
      return on.status === 'complete' && off.status === 'complete' ? [on.finalTestRmse - off.finalTestRmse] : [];
    }).sort((a, b) => a - b);
    const middle = Math.floor(differences.length / 2);
    const median = differences.length ? differences.length % 2 ? differences[middle] : (differences[middle - 1] + differences[middle]) / 2 : null;
    screenEffects.push({operation, scale, flag, comparedPairs: differences.length, divergentPairs: enabled.length - differences.length,
      lowerRmseWhenEnabled: differences.filter(value => value < -1e-6).length,
      higherRmseWhenEnabled: differences.filter(value => value > 1e-6).length,
      medianRmseDeltaEnabledMinusDisabled: median});
  }
const summary = {rotary: rotaryRows, scalar: {screenConfigurations: scalar.screenConfigurations, finalRuns: scalar.finalConfigurations,
  completeRuns: scalarRuns.filter(run => run.status === 'complete').length, divergentRuns: scalarRuns.filter(run => run.status === 'nonfinite').length,
  families, screenEffects, trainingFit, firstLayerGeometry: firstLayerRows}, validation: {...verification, pairedInitializations: initializations.size,
    savedPredictionsMatchMetrics: true, screeningPrefixesMatchFullRuns: true, initiallyDeadFirstLayerUnitsStayInactive: true}};
writeFileSync(join(base, 'comparison.json'), JSON.stringify(summary, null, 2) + '\n');
const writeCsv = (name, rows, fields) => writeFileSync(join(base, name), [fields.join(','), ...rows.map(row => fields.map(field => row[field] ?? '').join(','))].join('\n') + '\n');
writeCsv('rotary-runs.csv', rotary.runs, ['profile', 'seed', 'residual', 'learningRate', 'weightDecay', 'completedEpochs', 'firstTrain99', 'firstTest90', 'firstTest99', 'peakTest', 'finalTest', 'finalTrain', 'finalTestLoss']);
writeCsv('scalar-runs.csv', scalarRuns, ['stage', 'operation', 'scale', 'width', 'depth', 'preRmsnorm', 'residual', 'learningRate', 'weightDecay', 'seed', 'epochs', 'status', 'parameters', 'finalTrainMse', 'finalTestMse', 'finalTrainRmse', 'finalTestRmse', 'finalTrainAccuracy', 'finalTestAccuracy', 'constantTestRmse']);
const percent = value => (100 * value).toFixed(2) + '%';
const lines = ['LOCAL RUST/CUDA EXPERIMENTS', '', '1. W128 residual ablation',
  'Same matrices, initialization, split, minibatches, pre-block RMSNorm and final RMSNorm. Only the skip changes.',
  'Seeds: 11 / 17 / 23. Stop at first 99% held-out accuracy, otherwise 10,000 complete epochs.', ''];
for (const profile of rotaryRows) {
  lines.push(`${profile.name}: lr=${profile.learningRate}, wd=${profile.weightDecay}`,
    '  With skip, epochs to 99%: ' + profile.withResidual.map(run => run.firstTest99 ?? 'not reached').join(' / '),
    '  Without skip, epochs to 99%: ' + profile.withoutResidual.map(run => run.firstTest99 ?? 'not reached').join(' / '),
    '  Without skip, peak held-out accuracy: ' + profile.withoutResidual.map(run => percent(run.peakTest)).join(' / '), '');
}
lines.push('Residual is preferable for these existing profiles: all 9 controls reach 99%, versus 3/9 without skip.',
  'This paired ablation does not establish the best separately retuned no-skip optimizer.', '',
  '2. Two-float-input, scalar-output MSE MLP',
  'Division and multiplication mod 97; raw residues and unit-scaled inputs/targets (divide by 96).',
  'ReLU hidden layers; Kaiming-normal hidden weights; linear-gain output; zero initial biases.',
  'Pre-RMSNorm and residual apply only to width-preserving hidden layers; no input-pair norm or final norm.',
  'AdamW decays weights and biases. Batch 512 with a short tail; 10-update warmup; beta=(0.9,0.98).',
  `${scalar.screenConfigurations} seed-11 screens at ${scalar.plan.scalarMlp.screenEpochs} epochs, followed by ${scalar.finalConfigurations} selected runs at ${scalar.plan.scalarMlp.finalEpochs} epochs.`,
  'Depth 1 is a shared baseline with no eligible norm/skip layers. Full-budget selection minimizes final seed-11 screening RMSE within each family.',
  'The held-out split is used for selection, so this is exploratory validation, not an untouched test estimate.', '',
  'Final means below use all three seeds. RMSE is in original residue units; accuracy rounds without clipping or modulo reduction.', '');
for (const family of families) {
  const f = family.family, c = family.configuration;
  lines.push(`${f.operation}/${f.scale}, requested norm=${f.preRmsnorm}, skip=${f.residual}: W${c.width} D${c.depth}, actual norm=${c.preRmsnorm}, skip=${c.residual}, lr=${c.learningRate}, wd=${c.weightDecay}, ${c.parameters} parameters`,
    `  Train RMSE: ${family.seeds.map(run => run.trainRmse.toFixed(3)).join(' / ')}; mean ${family.meanTrainRmse?.toFixed(3) ?? 'divergence'}`,
    `  Test RMSE: ${family.seeds.map(run => run.testRmse.toFixed(3)).join(' / ')}; mean ${family.meanTestRmse?.toFixed(3) ?? 'divergence'}`,
    `  Rounded train accuracy: ${family.seeds.map(run => percent(run.trainAccuracy)).join(' / ')}; mean ${family.meanTrainAccuracy === null ? 'divergence' : percent(family.meanTrainAccuracy)}`,
    `  Rounded test accuracy: ${family.seeds.map(run => percent(run.testAccuracy)).join(' / ')}; mean ${family.meanTestAccuracy === null ? 'divergence' : percent(family.meanTestAccuracy)}`,
    `  Held-out accuracy on nonzero targets: ${family.seeds.map(run => run.heldoutTargetGroups ? percent(run.heldoutTargetGroups.nonzero.accuracy) : 'divergence').join(' / ')}`,
    `  Train-mean constant baseline, mean test RMSE: ${family.meanConstantTestRmse.toFixed(3)}`, '');
}
for (const [stage, fit] of Object.entries(trainingFit)) lines.push(`${stage}: best final train accuracy ${fit.bestFinal ? percent(fit.bestFinal.accuracy) : 'unavailable'}; highest logged train accuracy ${fit.bestObserved ? percent(fit.bestObserved.accuracy) + ' at epoch ' + fit.bestObserved.epoch : 'unavailable'}.`);
lines.push('');
const first128 = firstLayerRows.find(row => row.width === 128 && row.seed === 11 && row.operation === 'mul' && row.scale === 'unit');
if (first128) lines.push(`Initialization diagnostic: ${first128.initial.globallyDead}/128 first-layer ReLU units are inactive on the entire unit-scaled operand square at seed 11.`,
  'All initially inactive first-layer units remain inactive in every completed run. Centering is tested separately; this geometric observation alone does not establish the main cause of poor fit.', '');
lines.push('Paired seed-11 screen effects at fixed width, depth, lr, wd and the other flag:',
  'Negative enabled-minus-disabled RMSE means the enabled flag helped; these are short-screen results, not three-seed guarantees.');
for (const effect of screenEffects) lines.push(`${effect.operation}/${effect.scale}, ${effect.flag}: median RMSE delta ${effect.medianRmseDeltaEnabledMinusDisabled?.toFixed(4) ?? 'unavailable'}, helped in ${effect.lowerRmseWhenEnabled}/${effect.comparedPairs} finite pairs; ${effect.divergentPairs} divergent pairs.`);
lines.push('');
lines.push(`${summary.scalar.divergentRuns} divergent scalar runs. No interrupted runs count as negative results.`,
  'Scope: finite grid and budget; the regression and tied-embedding classifiers differ in architecture, representation and loss together.',
  'Training forward/backward and AdamW run on CUDA. Rust on the host aggregates saved metrics; f64 differentiation is verification only.',
  `Recorded verification: ${verification.rustTests} Rust tests; strict Clippy=${verification.strictClippy}; ${verification.rotaryCudaChecks} rotary and ${verification.scalarCudaChecks} scalar CUDA checks.`,
  verification.exactRotaryReplay ? 'Unchanged W128 replay matches original metrics and both checkpoints.' : 'No W128 replay record supplied.',
  verification.affineControl ? `End-to-end affine control: held-out MSE ${verification.affineControl.initialHeldoutMse} -> ${verification.affineControl.finalHeldoutMse}; passed=${verification.affineControl.passed}.` : 'No affine control record supplied.',
  'Evidence: comparison.json, rotary-runs.csv, scalar-runs.csv, per-trial arguments/manifests/histories/checkpoints, build source hashes and verification logs.',
  'This workflow writes local artifacts and performs no Git or publication changes.');
writeFileSync(join(base, 'report.txt'), lines.join('\n') + '\n');
console.log(`Validated ${scalarRuns.length} scalar histories, saved predictions and paired initializations; wrote compact comparison tables and report.`);
