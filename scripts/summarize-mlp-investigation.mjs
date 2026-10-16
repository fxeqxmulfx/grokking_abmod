// Audit native artifacts and combine the original and diagnostic MLP evidence.
// Usage: node scripts/summarize-mlp-investigation.mjs <evidence-root> [--partial]
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync, existsSync, readdirSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {createHash} from 'node:crypto';

const base = resolve(process.argv[2] ?? 'target/local-ablations-2026-10-16');
const partial = process.argv.includes('--partial');
const json = path => JSON.parse(readFileSync(path));
const sha = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const digest = buffers => {const hash = createHash('sha256'); for (const buffer of buffers) hash.update(buffer); return hash.digest('hex');};
const csv = path => {
  const [header, ...lines] = readFileSync(path, 'utf8').trim().split('\n'), fields = header.split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [fields[index], Number(value)])));
};
const metadata = path => Object.fromEntries(readFileSync(path, 'utf8').trim().split('\n').map(line => {
  const at = line.indexOf('='); return [line.slice(0, at), line.slice(at + 1)];
}));
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const median = values => {const sorted = values.toSorted((a, b) => a - b), mid = Math.floor(sorted.length / 2); return sorted.length ? sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2 : null;};
const rounded = value => Math.trunc(value + (value >= 0 ? .5 : -.5));
const percent = value => value === null || value === undefined ? 'unavailable' : (100 * value).toFixed(2) + '%';
const settings = run => ({operation: run.operation, modulus: run.modulus ?? 97, scale: run.scale, trainLimit: run.trainLimit ?? null,
  width: run.width, depth: run.depth, preRmsnorm: run.preRmsnorm, residual: run.residual, learningRate: run.learningRate,
  weightDecay: run.weightDecay, beta2: run.beta2 ?? .98, batch: run.batch ?? 512, schedule: run.schedule ?? 'constant',
  minLrRatio: run.minLrRatio ?? .001, biasInitialization: run.biasInitialization ?? 'zero', biasScale: run.biasScale ?? 1});
const key = run => JSON.stringify(settings(run));
const brief = run => {const s = settings(run); return `${s.operation}/p${s.modulus}/${s.scale}, N=${s.trainLimit ?? 'full'}, W${s.width} D${s.depth}, norm=${s.preRmsnorm}, skip=${s.residual}, bias=${s.biasInitialization}, lr=${s.learningRate}, wd=${s.weightDecay}, beta2=${s.beta2}, batch=${s.batch}, ${s.schedule}, ${run.epochs} epochs / ${run.updates} updates`;};
const required = ['scalar', 'mlp-controls', 'mlp-bias-screen', 'mlp-deep', 'mlp-fullbatch-pilot',
  'mlp-capacity-pilot', 'mlp-optimization-ablations', 'mlp-selected-confirmations'];
const names = [...new Set([...required, ...readdirSync(base, {withFileTypes: true}).filter(entry => entry.isDirectory() && entry.name.startsWith('mlp-') && existsSync(join(base, entry.name, 'source.json'))).map(entry => entry.name)])];
const phases = [], runs = [], histories = new Map(), matrices = new Map(), zeroInitializers = new Map(), laterBiases = new Map(), binaries = new Map(), exactTrials = new Map();
let replayedTrials = 0, savedPredictions = 0;
const prefixChecks = [];
const splitFingerprints = new Map();
const gridCoverage = [];
const cartesian = axes => Object.entries(axes).reduce((rows, [field, values]) => rows.flatMap(row => values.map(value => ({...row, [field]: value}))), [{}]);
const trialSignature = run => key(run) + '/' + run.seed + '/' + run.epochs;

function auditCoverage(name, own, source, directory) {
  const planPath = name === 'scalar' ? join(base, 'source/analysis/experiments/local-residual-regression.json') : join(directory, 'source-plan.json');
  assert.equal(sha(planPath), source.planSha256);
  const plan = json(planPath);
  assert.deepEqual(json(join(directory, 'plan.json')), plan, 'Saved interpreted plan differs from its raw source.');
  const common = {beta2: plan.optimizer.beta2, minLrRatio: .001, biasInitialization: 'zero', biasScale: 1, seed: plan.screenSeed};
  const checkedSet = (actual, expected, label) => {
    const wanted = new Map(expected.map(run => [trialSignature(run), run]));
    assert.equal(wanted.size, expected.length, 'Duplicate planned configuration: ' + label);
    assert.equal(new Set(actual.map(trialSignature)).size, actual.length, 'Duplicate recorded configuration: ' + label);
    for (const run of actual) {
      const expectedRun = wanted.get(trialSignature(run));
      assert.ok(expectedRun, 'Unplanned configuration: ' + label + ' ' + brief(run));
      if (expectedRun.reportEvery !== undefined) assert.equal(run.reportEvery, expectedRun.reportEvery, 'Wrong reporting interval.');
      if (expectedRun.variant !== undefined) assert.equal(run.variant, expectedRun.variant, 'Wrong ablation label.');
    }
    if (source.completed) assert.equal(actual.length, expected.length, 'Missing planned trials: ' + label);
    return {expected: expected.length, recorded: actual.length};
  };
  if (source.phase === 'followup') {
    const trials = json(join(directory, 'source-trials.json')).trials.map(run => ({...common, modulus: 97, trainLimit: null,
      preRmsnorm: false, residual: false, weightDecay: 0, schedule: 'cosine', batch: 512, reportEvery: 10, ...run}));
    assert.ok(own.every(run => run.stage === 'followup'));
    return {name, trials: checkedSet(own, trials, name), selectionsChecked: 0};
  }
  if (source.phase === 'controls') {
    const g = plan.controls;
    const expected = cartesian({dataset: g.datasets, operation: g.operations, scale: g.scales, flags: g.flags, learningRate: g.learningRates})
      .map(({dataset, flags, ...row}) => {
        const config = {...common, ...dataset, ...flags, ...row, width: g.width, depth: g.depth, weightDecay: g.weightDecay,
          schedule: g.schedule, minLrRatio: g.minimumLrRatio};
        const rows = config.trainLimit ?? Math.floor(config.modulus * (config.modulus - Number(config.operation === 'div')) / 2);
        config.batch = Math.min(g.batch, rows);
        const updatesPerEpoch = Math.ceil(rows / config.batch);
        assert.equal(g.updates % updatesPerEpoch, 0);
        config.epochs = g.updates / updatesPerEpoch;
        return config;
      });
    assert.ok(own.every(run => run.stage === 'control'));
    return {name, trials: checkedSet(own, expected, name), selectionsChecked: 0};
  }
  assert.ok(name === 'scalar' || source.phase === 'deep');
  const original = name === 'scalar', g = original ? plan.scalarMlp : plan.deep;
  const screening = cartesian({width: g.widths, depth: g.depths, preRmsnorm: g.preRmsnorm, residual: g.residual,
    learningRate: g.learningRates, weightDecay: g.weightDecays, operation: g.operations, scale: g.scales})
    .filter(run => run.depth !== 1 || !run.preRmsnorm && !run.residual)
    .map(run => ({...common, ...run, modulus: original ? plan.modulus : g.modulus, trainLimit: null,
      batch: original ? plan.batch : g.batch, schedule: original ? 'constant' : g.schedule,
      seed: original ? g.screenSeed : plan.screenSeed, epochs: g.screenEpochs, reportEvery: original ? g.screenReportEveryEpochs : 10}));
  assert.ok(own.every(run => ['screen', 'final'].includes(run.stage)));
  const screens = own.filter(run => run.stage === 'screen'), finals = own.filter(run => run.stage === 'final');
  const coverage = {name, screens: checkedSet(screens, screening, name + '/screen'), finals: {expected: null, recorded: finals.length}, selectionsChecked: 0};
  const selectionPath = join(directory, 'finalists.json');
  if (!existsSync(selectionPath)) {
    assert.ok(!source.completed && finals.length === 0, 'Missing selection evidence.');
    return coverage;
  }
  assert.equal(screens.length, screening.length, 'Selected finalists before completing the screening grid.');
  const selected = json(selectionPath), winners = [];
  const choose = (candidates, metric, accuracy) => candidates.filter(run => run.status === 'complete')
    .toSorted((a, b) => a[metric] - b[metric] || b[accuracy] - a[accuracy] || a.parameters - b.parameters)[0];
  if (original) {
    const families = cartesian({operation: g.operations, scale: g.scales, preRmsnorm: g.preRmsnorm, residual: g.residual});
    assert.equal(selected.length, families.length);
    for (const family of families) {
      const matches = selected.filter(selection => Object.keys(family).every(field => selection.family[field] === family[field]));
      assert.equal(matches.length, 1);
      const eligible = screens.filter(run => run.operation === family.operation && run.scale === family.scale
        && (run.depth === 1 || run.preRmsnorm === family.preRmsnorm && run.residual === family.residual));
      const winner = choose(eligible, 'finalTestRmse', 'finalTestAccuracy');
      assert.ok(winner);
      assert.equal(trialSignature(matches[0].configuration), trialSignature(winner), 'Original finalist does not match the selection rule.');
      winners.push(winner);
    }
  } else {
    const tasks = cartesian({operation: g.operations, scale: g.scales, criterion: ['training-fit', 'held-out-fit']});
    assert.equal(selected.length, tasks.length);
    for (const task of tasks) {
      const matches = selected.filter(selection => Object.keys(task).every(field => selection[field] === task[field]));
      assert.equal(matches.length, 1);
      const training = task.criterion === 'training-fit';
      const winner = choose(screens.filter(run => run.operation === task.operation && run.scale === task.scale),
        training ? 'finalTrainRmse' : 'finalTestRmse', training ? 'finalTrainAccuracy' : 'finalTestAccuracy');
      assert.ok(winner);
      assert.equal(trialSignature(matches[0].screen), trialSignature(winner), 'Deep finalist does not match the selection rule.');
      winners.push(winner);
    }
  }
  const expectedFinals = [...new Map(winners.flatMap(winner => plan.seeds.map(seed => ({...winner, seed, epochs: g.finalEpochs,
    reportEvery: original ? g.finalReportEveryEpochs : 100}))).map(run => [trialSignature(run), run])).values()];
  coverage.finals = checkedSet(finals, expectedFinals, name + '/final');
  coverage.selectionsChecked = winners.length;
  return coverage;
}

function checkInitialization(buffer, run) {
  let offset = 0;
  const matrixParts = [], laterBiasParts = [], allBiasParts = [];
  for (let index = 0; index <= run.depth; index++) {
    const inputs = index === 0 ? 2 : run.width, outputs = index === run.depth ? 1 : run.width;
    const end = offset + inputs * outputs * 4;
    matrixParts.push(buffer.subarray(offset, end));
    allBiasParts.push(buffer.subarray(end, end + outputs * 4));
    if (index > 0) laterBiasParts.push(buffer.subarray(end, end + outputs * 4));
    offset = end + outputs * 4;
  }
  assert.equal(offset, buffer.length, 'Initializer layout differs from the architecture.');
  const shape = `${run.width}/${run.depth}/${run.seed}`, matrixSha = digest(matrixParts);
  if (matrices.has(shape)) assert.equal(matrixSha, matrices.get(shape), 'Paired Kaiming matrices changed.');
  matrices.set(shape, matrixSha);
  const mode = settings(run).biasInitialization;
  if (mode === 'zero') {
    for (const biases of allBiasParts) for (let at = 0; at < biases.length; at += 4) assert.equal(biases.readFloatLE(at), 0);
    const hash = digest([buffer]);
    if (zeroInitializers.has(shape)) assert.equal(hash, zeroInitializers.get(shape), 'Paired zero-bias initialization changed.');
    zeroInitializers.set(shape, hash);
  } else {
    const pair = shape + '/' + settings(run).biasScale, hash = digest(laterBiasParts);
    if (laterBiases.has(pair)) assert.equal(hash, laterBiases.get(pair), 'Uniform/domain later biases changed.');
    laterBiases.set(pair, hash);
  }
}

function validate(run, source) {
  const s = settings(run), directory = run.directory, manifest = metadata(join(directory, 'run.txt'));
  assert.ok(['complete', 'nonfinite'].includes(run.status));
  assert.equal(manifest.status, run.status);
  assert.equal(Number(manifest.modulus), s.modulus);
  assert.equal(manifest.operation, s.operation);
  assert.equal(manifest.scale, s.scale);
  assert.equal(Number(manifest.width), s.width);
  assert.equal(Number(manifest.depth), s.depth);
  assert.equal(manifest.pre_rmsnorm, String(s.preRmsnorm));
  assert.equal(manifest.residual, String(s.residual));
  assert.equal(Number(manifest.seed), run.seed);
  assert.equal(Number(manifest.requested_epochs), run.epochs);
  assert.equal(Number(manifest.batch), s.batch);
  assert.equal(Number(manifest.train_fraction), .5);
  assert.equal(manifest.bias_initialization ?? 'zero', s.biasInitialization);
  for (const [field, expected] of Object.entries({learning_rate: s.learningRate, weight_decay: s.weightDecay,
    minimum_lr_ratio: s.minLrRatio, bias_initialization_scale: s.biasScale})) {
    if (['learning_rate', 'weight_decay'].includes(field)) assert.ok(field in manifest, 'Missing native ' + field);
    const actual = Number(manifest[field] ?? expected);
    assert.ok(Math.abs(actual - expected) <= 1e-6 * Math.max(.001, Math.abs(expected)), 'Wrong native ' + field);
  }
  assert.equal(manifest.schedule ?? 'constant', s.schedule);
  assert.deepEqual(manifest.betas.split(',').map(Number), [.9, s.beta2]);
  const outputScale = s.scale === 'raw' ? 1 : s.modulus - 1;
  assert.equal(Number(manifest.output_scale), outputScale);
  assert.equal(Number(manifest.output_offset ?? 0), s.scale === 'centered' ? .5 : 0);
  const total = s.modulus * (s.modulus - Number(s.operation === 'div')), trainingPool = Math.floor(total / 2);
  const trainRows = s.trainLimit ?? trainingPool, heldoutRows = total - trainingPool;
  assert.equal(Number(manifest.training_equations), trainRows);
  assert.equal(Number(manifest.validation_equations), heldoutRows);
  if (manifest.training_pool_equations !== undefined) assert.equal(Number(manifest.training_pool_equations), trainingPool);
  if (manifest.unused_training_equations !== undefined) assert.equal(Number(manifest.unused_training_equations), trainingPool - trainRows);
  const arguments_ = json(join(directory, 'arguments.json'));
  if (!binaries.has(arguments_.binary)) binaries.set(arguments_.binary, sha(arguments_.binary));
  assert.equal(binaries.get(arguments_.binary), source.binarySha256);
  assert.equal(sha(join(directory, 'loss.csv')), run.historySha256);
  const initial = readFileSync(join(directory, 'initial-model.bin'));
  assert.equal(initial.length, run.parameters * 4);
  checkInitialization(initial, run);
  const history = csv(join(directory, 'loss.csv')), interval = Number(manifest.report_every_epochs), updates = Math.ceil(trainRows / s.batch);
  if (run.reportEvery !== undefined) assert.equal(interval, run.reportEvery);
  assert.ok(history.length);
  for (const [index, row] of history.entries()) {
    assert.ok(Object.values(row).every(Number.isFinite));
    assert.equal(row.epoch, Math.min(index * interval, run.epochs));
    assert.equal(row.step, row.epoch * updates);
  }
  histories.set(directory, history);
  const normalized = history.map(({elapsed_seconds, ...row}) => row);
  const replayKey = `${key(run)}/${run.seed}/${run.epochs}/${interval}`;
  if (exactTrials.has(replayKey)) {assert.deepEqual(normalized, exactTrials.get(replayKey), 'Repeated native trial changed metrics.'); replayedTrials++;}
  exactTrials.set(replayKey, normalized);
  const peakTrainRow = history.reduce((best, row) => row.train_accuracy > best.train_accuracy ? row : best);
  const result = {...run, ...s, trainingRows: trainRows, heldoutRows, reportEvery: interval,
    updates: Number(manifest.updates), peakTrainAccuracy: Math.max(...history.map(row => row.train_accuracy)),
    peakTrainEpoch: peakTrainRow.epoch,
    peakTestAccuracy: Math.max(...history.map(row => row.test_accuracy)), predictionStatistics: null};
  if (run.status !== 'complete') return result;
  assert.equal(history.at(-1).epoch, run.epochs);
  assert.equal(readFileSync(join(directory, 'model.bin')).length, run.parameters * 4);
  const predictions = csv(join(directory, 'predictions.csv'));
  assert.equal(predictions.length, heldoutRows);
  const pairs = new Set(), groups = {zero: {rows: 0, correct: 0}, nonzero: {rows: 0, correct: 0}}, equations = createHash('sha256');
  let squares = 0, correct = 0, predictedSum = 0, targetSum = 0;
  for (const row of predictions) {
    assert.ok([row.a, row.b, row.target].every(value => Number.isInteger(value) && value >= 0 && value < s.modulus));
    assert.ok(Number.isFinite(row.prediction_residue));
    const pair = `${row.a}/${row.b}`;
    assert.ok(!pairs.has(pair), 'Repeated held-out equation.'); pairs.add(pair);
    if (s.operation === 'div') assert.ok(row.b > 0 && row.target * row.b % s.modulus === row.a);
    else {assert.equal(s.operation, 'mul'); assert.equal(row.target, row.a * row.b % s.modulus);}
    equations.update(`${row.a},${row.b},${row.target}\n`);
    const hit = Number(rounded(row.prediction_residue) === row.target), group = groups[row.target === 0 ? 'zero' : 'nonzero'];
    group.rows++; group.correct += hit; correct += hit;
    squares += (row.prediction_residue - row.target) ** 2;
    predictedSum += row.prediction_residue; targetSum += row.target;
  }
  const rmse = Math.sqrt(squares / heldoutRows), accuracy = correct / heldoutRows;
  const splitKey = `${s.operation}/${s.modulus}/${run.seed}`, splitHash = equations.digest('hex');
  if (splitFingerprints.has(splitKey)) assert.equal(splitHash, splitFingerprints.get(splitKey), 'Held-out equations changed across paired configurations.');
  splitFingerprints.set(splitKey, splitHash);
  assert.ok(Math.abs(rmse - run.finalTestRmse) <= 2e-6 * Math.max(1, rmse));
  assert.ok(Math.abs(accuracy - run.finalTestAccuracy) < 1e-8);
  assert.ok(Math.abs(rmse ** 2 - history.at(-1).test_mse * outputScale ** 2) <= 1e-5 * Math.max(1, rmse ** 2), 'Training-unit MSE and restored predictions disagree.');
  const predictionMean = predictedSum / heldoutRows, targetMean = targetSum / heldoutRows;
  result.predictionStatistics = {rmse, accuracy, predictionMean, targetMean,
    predictionStddev: Math.sqrt(mean(predictions.map(row => (row.prediction_residue - predictionMean) ** 2))),
    targetStddev: Math.sqrt(mean(predictions.map(row => (row.target - targetMean) ** 2))),
    targetGroups: Object.fromEntries(Object.entries(groups).map(([name, group]) => [name, {...group, accuracy: group.rows ? group.correct / group.rows : null}]))};
  savedPredictions += heldoutRows;
  return result;
}

for (const name of names) {
  const directory = join(base, name), sourcePath = join(directory, 'source.json');
  if (!existsSync(sourcePath)) {assert.ok(partial, 'Missing required phase: ' + name); phases.push({name, status: 'pending', runs: 0}); continue;}
  const source = json(sourcePath);
  assert.ok(partial || source.completed, 'Phase is not complete: ' + name);
  if (name !== 'scalar') {
    assert.equal(sha(join(directory, 'source-plan.json')), source.planSha256);
    assert.equal(sha(join(directory, 'source-driver.mjs')), source.driverSha256);
    if (source.trialsSha256) assert.equal(sha(join(directory, 'source-trials.json')), source.trialsSha256);
  }
  const own = existsSync(join(directory, 'runs.json')) ? json(join(directory, 'runs.json')) : [];
  const checked = own.map(run => ({...validate(run, source), phase: name}));
  runs.push(...checked);
  gridCoverage.push(auditCoverage(name, checked, source, directory));
  phases.push({name, status: source.completed ? 'complete' : 'running', runs: own.length,
    reason: existsSync(join(directory, 'source-trials.json')) ? json(join(directory, 'source-trials.json')).reason : null,
    complete: own.filter(run => run.status === 'complete').length, divergent: own.filter(run => run.status === 'nonfinite').length});
}

// Reloaded checkpoints must reproduce both splits, including the reported train fit.
const checkpointReplays = new Map();
for (const entry of readdirSync(base, {withFileTypes: true}).filter(entry => entry.isDirectory() && entry.name.startsWith('checkpoint-verification-'))) {
  const path = join(base, entry.name, 'summary.json');
  if (!existsSync(path)) {assert.ok(partial, 'Incomplete checkpoint verification: ' + entry.name); continue;}
  const summary = json(path);
  assert.equal(summary.passed, true);
  assert.equal(sha(summary.binary), summary.evaluatorSha256);
  assert.equal(summary.evaluatorSha256, json(join(base, 'evaluator-build-source.json')).binarySha256);
  for (const record of summary.records) {
    assert.equal(record.passed, true);
    const run = runs.find(run => run.directory === record.directory);
    assert.ok(run, 'Verified checkpoint is absent from the recorded experiments.');
    for (const [file, expected] of Object.entries(record.inputSha256)) assert.equal(sha(join(record.directory, file)), expected);
    for (const [file, expected] of Object.entries(record.predictionSha256)) assert.equal(sha(join(record.evaluationDirectory, file)), expected);
    assert.equal(sha(join(record.evaluationDirectory, 'stdout.json')), record.stdoutSha256);
    assert.deepEqual(json(join(record.evaluationDirectory, 'stdout.json')), record.native);
    const final = histories.get(run.directory).at(-1);
    for (const split of ['train', 'test']) for (const field of ['mse', 'rmse', 'accuracy']) {
      const expected = final[`${split}_${field === 'rmse' ? 'rmse_residue' : field}`], actual = record.native[split][field];
      assert.ok(Math.abs(actual - expected) <= (field === 'mse' ? 6e-13 : 6e-10), 'Checkpoint replay disagrees with its training history.');
      assert.ok(Math.abs(actual - record.independentlyRecomputed[split][field]) <= 1e-10 * Math.max(1, actual));
    }
    assert.ok(!checkpointReplays.has(run.directory), 'Duplicate checkpoint verification.');
    checkpointReplays.set(run.directory, {verification: path, directory: record.evaluationDirectory,
      modelSha256: record.inputSha256['model.bin'], train: record.native.train, test: record.native.test});
  }
}

const selectedConfirmations = [];
for (const phase of phases.filter(phase => existsSync(join(base, phase.name, 'source-trials.json')))) {
  const plan = json(join(base, phase.name, 'source-trials.json'));
  if (!plan.reuseSeed11) continue;
  assert.deepEqual(plan.confirmationSeeds, [11, 17, 23]);
  assert.equal(new Set(plan.reuseSeed11.map(reference => reference.operation)).size, plan.reuseSeed11.length);
  for (const reference of plan.reuseSeed11) {
    const run = runs.find(run => run.directory === resolve(base, reference.directory));
    assert.ok(run && run.status === 'complete', 'Missing conclusive reused seed-11 result.');
    assert.equal(run.seed, 11); assert.equal(reference.seed, 11); assert.equal(run.operation, reference.operation);
    assert.equal(sha(join(run.directory, 'loss.csv')), reference.historySha256);
    assert.equal(sha(join(run.directory, 'model.bin')), reference.modelSha256);
    assert.equal(sha(join(run.directory, 'initial-model.bin')), reference.initializerSha256);
    assert.equal(run.finalTrainAccuracy, reference.finalTrainAccuracy);
    assert.equal(run.finalTestAccuracy, reference.finalTestAccuracy);
    const planned = plan.trials.filter(trial => trial.operation === reference.operation);
    assert.deepEqual(planned.map(trial => trial.seed).toSorted((a, b) => a - b), [17, 23]);
    for (const trial of planned) {assert.equal(key(trial), key(run), 'Confirmation changed the selected configuration.'); assert.equal(trial.epochs, run.epochs);}
    const own = [run, ...runs.filter(candidate => candidate.phase === phase.name && candidate.operation === reference.operation)];
    assert.equal(new Set(own.map(run => run.seed)).size, own.length);
    if (phase.status === 'complete') assert.deepEqual(own.map(run => run.seed).toSorted((a, b) => a - b), plan.confirmationSeeds);
    const complete = own.length === 3 && own.every(run => run.status === 'complete');
    for (const candidate of own.filter(run => run.status === 'complete')) assert.ok(partial || checkpointReplays.has(candidate.directory), 'Selected trained model still needs a checkpoint replay.');
    selectedConfirmations.push({phase: phase.name, configuration: settings(run), epochs: run.epochs, seeds: plan.confirmationSeeds,
      complete, runs: own.map(run => ({seed: run.seed, directory: run.directory, status: run.status,
        trainAccuracy: run.finalTrainAccuracy, testAccuracy: run.finalTestAccuracy, trainRmse: run.finalTrainRmse,
        testRmse: run.finalTestRmse, checkpointReplayed: checkpointReplays.has(run.directory)}))});
  }
  assert.equal(plan.trials.length, plan.reuseSeed11.length * 2);
}

// Deep confirmations report less frequently; compare every shared metric row.
for (const run of runs.filter(run => ['scalar', 'mlp-deep'].includes(run.phase) && run.stage === 'final' && run.seed === 11)) {
  const screen = runs.find(candidate => candidate.phase === run.phase && candidate.stage === 'screen' && candidate.seed === run.seed && key(candidate) === key(run));
  assert.ok(screen);
  const earlier = new Map(histories.get(screen.directory).map(({elapsed_seconds, ...row}) => [row.epoch, row]));
  let sharedRows = 0;
  for (const {elapsed_seconds, ...row} of histories.get(run.directory)) if (earlier.has(row.epoch)) {
    assert.deepEqual(row, earlier.get(row.epoch), 'Confirmation changed the screening prefix.'); sharedRows++;
  }
  assert.ok(sharedRows > 0);
  prefixChecks.push({directory: run.directory, screenDirectory: screen.directory, sharedMetricRows: sharedRows});
}
const groups = new Map();
for (const run of runs.filter(run => run.modulus === 97 && run.trainLimit === null && run.epochs >= 10000)) {
  const groupKey = key(run) + '/' + run.epochs;
  if (!groups.has(groupKey)) groups.set(groupKey, new Map());
  groups.get(groupKey).set(run.seed, run);
}
const threeSeed = [...groups.values()].filter(group => [11, 17, 23].every(seed => group.has(seed))).map(group => {
  const own = [11, 17, 23].map(seed => group.get(seed)), complete = own.every(run => run.status === 'complete');
  return {configuration: settings(own[0]), epochs: own[0].epochs, parameters: own[0].parameters, complete, runs: own,
    meanTrainRmse: complete ? mean(own.map(run => run.finalTrainRmse)) : null,
    meanTestRmse: complete ? mean(own.map(run => run.finalTestRmse)) : null,
    meanTrainAccuracy: complete ? mean(own.map(run => run.finalTrainAccuracy)) : null,
    meanTestAccuracy: complete ? mean(own.map(run => run.finalTestAccuracy)) : null};
});
const best = (operation, scale, field, direction = 1) => threeSeed.filter(group => group.complete && group.configuration.operation === operation && group.configuration.scale === scale).toSorted((a, b) => direction * (a[field] - b[field]))[0] ?? null;
const bestByTask = ['div', 'mul'].flatMap(operation => ['unit', 'raw', 'centered'].map(scale => ({operation, scale,
  trainingFit: best(operation, scale, 'meanTrainRmse'), trainingAccuracyFit: best(operation, scale, 'meanTrainAccuracy', -1),
  heldoutFit: best(operation, scale, 'meanTestRmse'), heldoutAccuracyFit: best(operation, scale, 'meanTestAccuracy', -1)})));
const controlGroups = new Map();
for (const run of runs.filter(run => run.phase === 'mlp-controls' && run.status === 'complete')) {
  const label = `${run.operation}/p${run.modulus}/N${run.trainLimit ?? 'full'}/${run.scale}`;
  if (!controlGroups.has(label) || run.finalTrainRmse < controlGroups.get(label).finalTrainRmse) controlGroups.set(label, run);
}
const biasEffects = [];
for (const operation of ['div', 'mul']) for (const scale of ['unit', 'raw', 'centered']) {
  const paired = runs.filter(run => run.phase === 'mlp-bias-screen' && run.operation === operation && run.scale === scale && run.biasInitialization === 'domain');
  const differences = [];
  for (const domain of paired) {
    const uniform = runs.find(run => run.phase === domain.phase && run.operation === operation && run.scale === scale && run.learningRate === domain.learningRate && run.biasInitialization === 'uniform');
    assert.ok(uniform);
    if (domain.status === 'complete' && uniform.status === 'complete') differences.push(domain.finalTrainRmse - uniform.finalTrainRmse);
  }
  biasEffects.push({operation, scale, finitePairs: differences.length, medianTrainRmseDomainMinusUniform: median(differences)});
}
const deepEffects = [];
const deepScreens = runs.filter(run => run.phase === 'mlp-deep' && run.stage === 'screen');
for (const operation of ['div', 'mul']) for (const scale of ['unit', 'raw']) for (const flag of ['preRmsnorm', 'residual']) {
  const pairKey = run => {const s = settings(run); delete s[flag]; return JSON.stringify(s) + '/' + run.seed + '/' + run.epochs;};
  const off = new Map(deepScreens.filter(run => !run[flag]).map(run => [pairKey(run), run]));
  const train = [], test = [], accuracy = [];
  let divergentPairs = 0, pendingPairs = 0;
  for (const on of deepScreens.filter(run => run.operation === operation && run.scale === scale && run[flag])) {
    const control = off.get(pairKey(on));
    if (!control) {assert.ok(partial); pendingPairs++; continue;}
    if (on.status !== 'complete' || control.status !== 'complete') {divergentPairs++; continue;}
    train.push(on.finalTrainRmse - control.finalTrainRmse);
    test.push(on.finalTestRmse - control.finalTestRmse);
    accuracy.push(on.finalTrainAccuracy - control.finalTrainAccuracy);
  }
  deepEffects.push({operation, scale, flag, finitePairs: train.length, divergentPairs, pendingPairs,
    medianTrainRmseEnabledMinusDisabled: median(train), medianTestRmseEnabledMinusDisabled: median(test), medianTrainAccuracyEnabledMinusDisabled: median(accuracy)});
}
const changedField = {'width256': 'width', 'depth4': 'depth', 'zero-bias': 'biasInitialization', 'uniform-bias': 'biasInitialization',
  'beta2-098': 'beta2', 'constant-lr': 'schedule', 'batch512': 'batch'};
const optimizationAblations = [];
for (const run of runs.filter(run => ['mlp-optimization-ablations', 'mlp-capacity-pilot'].includes(run.phase))) {
  const reference = runs.find(candidate => candidate.phase === 'mlp-fullbatch-pilot' && candidate.operation === run.operation && candidate.seed === run.seed && candidate.epochs === run.epochs);
  if (!reference) {assert.ok(partial); continue;}
  const actual = settings(run), expected = settings(reference);
  const differences = Object.keys(actual).filter(field => JSON.stringify(actual[field]) !== JSON.stringify(expected[field]));
  assert.deepEqual(differences, [changedField[run.variant]], 'A one-factor comparison changed additional settings.');
  const initial = readFileSync(join(run.directory, 'initial-model.bin')), referenceInitial = readFileSync(join(reference.directory, 'initial-model.bin'));
  if (['beta2', 'schedule', 'batch'].includes(differences[0])) {
    assert.deepEqual(initial, referenceInitial, 'Optimizer comparison changed initialization.');
    const strip = ({elapsed_seconds, ...row}) => row;
    assert.deepEqual(strip(histories.get(run.directory)[0]), strip(histories.get(reference.directory)[0]), 'Optimizer comparison changed initial evaluation.');
  } else if (differences[0] === 'depth') {
    const sharedHidden = 3 * run.width + (Math.min(run.depth, reference.depth) - 1) * (run.width ** 2 + run.width);
    assert.deepEqual(initial.subarray(0, 4 * sharedHidden), referenceInitial.subarray(0, 4 * sharedHidden), 'Depth comparison changed shared hidden initializers.');
  }
  const complete = run.status === 'complete' && reference.status === 'complete';
  optimizationAblations.push({variant: run.variant, operation: run.operation, seed: run.seed, changedField: differences[0], status: run.status,
    directory: run.directory, referenceDirectory: reference.directory, epochs: run.epochs, updates: run.updates, referenceUpdates: reference.updates,
    trainAccuracy: complete ? run.finalTrainAccuracy : null, testAccuracy: complete ? run.finalTestAccuracy : null,
    trainRmse: complete ? run.finalTrainRmse : null, testRmse: complete ? run.finalTestRmse : null,
    trainAccuracyDelta: complete ? run.finalTrainAccuracy - reference.finalTrainAccuracy : null,
    trainRmseDelta: complete ? run.finalTrainRmse - reference.finalTrainRmse : null,
    testRmseDelta: complete ? run.finalTestRmse - reference.finalTestRmse : null});
}
const fullTask = runs.filter(run => run.modulus === 97 && run.trainLimit === null);
const completeTask = fullTask.filter(run => run.status === 'complete');
const highestFinal = completeTask.toSorted((a, b) => b.finalTrainAccuracy - a.finalTrainAccuracy)[0] ?? null;
const highestObserved = fullTask.toSorted((a, b) => b.peakTrainAccuracy - a.peakTrainAccuracy)[0] ?? null;
const construction = existsSync(join(base, 'scalar-arithmetic-control.jsonl')) ? readFileSync(join(base, 'scalar-arithmetic-control.jsonl'), 'utf8').trim().split('\n').map(JSON.parse) : [];
assert.ok(partial || construction.length === 2, 'Missing exhaustive representability control.');
for (const record of construction) {assert.equal(record.control, 'constructed_weights_not_trained'); assert.equal(record.passed, true); assert.equal(record.trainCorrect, record.trainRows); assert.equal(record.heldoutCorrect, record.heldoutRows); assert.equal(record.maxAbsoluteError, 0);}
const rotary = json(join(base, 'rotary/summary.json'));
assert.equal(json(join(base, 'rotary/source.json')).completed, true);
const rotaryExpected = rotary.plan.profiles.length * rotary.seeds.length * 2;
assert.equal(rotary.runs.length, rotaryExpected, 'Missing W128 paired trials.');
assert.equal(new Set(rotary.runs.map(run => `${run.profile}/${run.seed}/${run.residual}`)).size, rotaryExpected, 'Duplicate W128 trial.');
for (const run of rotary.runs) assert.equal(sha(join(run.directory, 'loss.csv')), run.historySha256);
const rotaryComparison = rotary.plan.profiles.map(profile => {
  const withSkip = rotary.seeds.map(seed => rotary.runs.find(run => run.profile === profile.name && run.seed === seed && run.residual));
  const withoutSkip = rotary.seeds.map(seed => rotary.runs.find(run => run.profile === profile.name && run.seed === seed && !run.residual));
  for (const [index, control] of withSkip.entries()) assert.equal(sha(join(control.directory, 'initial-model.bin')), sha(join(withoutSkip[index].directory, 'initial-model.bin')));
  return {...profile, withResidualEpochs99: withSkip.map(run => run.firstTest99), withoutResidualEpochs99: withoutSkip.map(run => run.firstTest99),
    withoutResidualTrainAccuracy: withoutSkip.map(run => run.finalTrain), withoutResidualTestAccuracy: withoutSkip.map(run => run.finalTest)};
});
const result = {status: partial ? 'in_progress' : 'complete', phases, runs, threeSeed, bestByTask, selectedConfirmations,
  bestTrainingControls: [...controlGroups.values()], biasEffects, deepEffects, optimizationAblations, highestFinal, highestObserved, constructedNotTrained: construction, rotaryComparison,
  validation: {histories: runs.length, savedPredictionRows: savedPredictions, matrixInitializations: matrices.size, pairedZeroInitializers: zeroInitializers.size,
    pairedLaterBiases: laterBiases.size, pairedHeldoutSplits: Object.fromEntries(splitFingerprints), identicalRepeatedTrials: replayedTrials, prefixChecks, gridCoverage,
    checkpointReplays: Object.fromEntries(checkpointReplays), nativeBinaries: Object.fromEntries(binaries)}};
const stem = partial ? 'mlp-investigation-partial' : 'mlp-investigation';
writeFileSync(join(base, stem + '.json'), JSON.stringify(result, null, 2) + '\n');
const lines = [partial ? 'MLP INVESTIGATION: IN PROGRESS' : 'MLP INVESTIGATION: COMPLETE', '',
  'Local Rust/CUDA training. Two numeric inputs, ReLU/Kaiming hidden layers, one linear output, MSE.',
  'Accuracy rounds restored scalar predictions without clipping or modulo reduction. RMSE uses original residue units.',
  'Each seed changes the train/held-out partition, parameter initialization and epoch order together; cross-seed variation does not isolate initialization alone.',
  'Held-out data are used for configuration selection; these are exploratory validation results.', ''];
lines.push('FINDINGS AND INTERPRETATION');
const originalFinite = runs.filter(run => run.phase === 'scalar' && run.status === 'complete');
if (originalFinite.length) lines.push(`The original numeric grid reached at most ${percent(Math.max(...originalFinite.map(run => run.finalTrainAccuracy)))} final train accuracy among its completed trials.`);
for (const cohort of selectedConfirmations.filter(cohort => cohort.complete)) {
  const train = cohort.runs.map(run => run.trainAccuracy), test = cohort.runs.map(run => run.testAccuracy), config = cohort.configuration;
  lines.push(`${config.operation}, selected W${config.width} D${config.depth}: three-seed mean train/held-out ${percent(mean(train))}/${percent(mean(test))}; train range ${percent(Math.min(...train))} to ${percent(Math.max(...train))}.`);
}
if (selectedConfirmations.some(cohort => cohort.complete)) lines.push('The selected learned models show substantial train memorization with weak held-out performance. Their measured seed variation includes changes in both the data split and initialization.');
for (const operation of ['div', 'mul']) {
  const group = threeSeed.filter(group => group.complete && group.configuration.operation === operation)
    .toSorted((a, b) => b.meanTestAccuracy - a.meanTestAccuracy)[0];
  if (group) lines.push(`${operation}: highest confirmed three-seed mean held-out rounded accuracy ${percent(group.meanTestAccuracy)} (train ${percent(group.meanTrainAccuracy)}); ${brief(group.runs[0])}.`);
}
for (const [variant, label] of [['depth4', 'Four hidden layers instead of eight'], ['beta2-098', 'Beta2 0.98 instead of 0.999'],
  ['constant-lr', 'Constant learning rate instead of cosine decay'], ['batch512', 'Batch 512 instead of a full training batch']]) {
  const paired = optimizationAblations.filter(row => row.variant === variant && row.status === 'complete');
  if (paired.length !== 2) continue;
  lines.push(`${label}, matched seed-11 controls: ` + paired.map(row => {
    const reference = runs.find(run => run.directory === row.referenceDirectory);
    return `${row.operation} train ${percent(row.trainAccuracy)} versus reference ${percent(reference.finalTrainAccuracy)}`;
  }).join('; ') + '.');
}
if (optimizationAblations.some(row => row.variant === 'batch512')) lines.push('Batch comparisons retain the same number of epochs and visits per equation; changing batch size also changes update frequency, so this comparison does not isolate gradient noise alone.');
const withSkipHits = rotaryComparison.flatMap(profile => profile.withResidualEpochs99).filter(value => value !== null).length;
const withoutSkipHits = rotaryComparison.flatMap(profile => profile.withoutResidualEpochs99).filter(value => value !== null).length;
lines.push(`W128 classifier: held-out 99% was reached in ${withSkipHits}/${rotary.runs.length / 2} existing-profile trials with the skip and ${withoutSkipHits}/${rotary.runs.length / 2} without it. This comparison retains both normalizations and uses the existing optimizer profiles.`);
if (construction.length === 2) lines.push('The exhaustive hand-set ReLU construction establishes representational capacity for the full arithmetic tables. It supplies a capacity control separately from all Kaiming-trained performance results.');
lines.push('Conclusions apply to these finite search spaces and budgets. A claim about MSE alone would require holding numeric representation and architecture fixed in a loss comparison.', '', 'PHASE STATUS');
for (const phase of phases) {
  lines.push(`${phase.name}: ${phase.status}, ${phase.runs} recorded trials, ${phase.divergent ?? 0} divergent.`);
  if (partial && phase.reason) lines.push('  Rationale recorded before launch: ' + phase.reason);
}
lines.push('', 'FULL MOD-97 TRAINING FIT (subsets and smaller moduli excluded)',
  highestFinal ? `Highest final rounded train accuracy: ${percent(highestFinal.finalTrainAccuracy)}; held-out ${percent(highestFinal.finalTestAccuracy)}; ${brief(highestFinal)}, seed ${highestFinal.seed}.` : 'No completed full mod-97 training trial.',
  highestObserved ? `Highest logged train accuracy: ${percent(highestObserved.peakTrainAccuracy)} at epoch ${highestObserved.peakTrainEpoch}; ${brief(highestObserved)}, seed ${highestObserved.seed}.` : 'No logged full-task result.', '',
  'THREE-SEED FULL-BUDGET RESULTS (means on seeds 11/17/23)');
for (const task of bestByTask) {
  lines.push(`${task.operation}/${task.scale}:`);
  for (const [label, group] of [['Best training RMSE', task.trainingFit], ['Best rounded training accuracy', task.trainingAccuracyFit],
    ['Best held-out RMSE', task.heldoutFit], ['Best rounded held-out accuracy', task.heldoutAccuracyFit]]) {
    if (!group) {lines.push(`  ${label}: no complete three-seed confirmation yet.`); continue;}
    lines.push(`  ${label}: ${brief(group.runs[0])}`,
      `    Train/test accuracy ${percent(group.meanTrainAccuracy)} / ${percent(group.meanTestAccuracy)}; residue RMSE ${group.meanTrainRmse.toFixed(3)} / ${group.meanTestRmse.toFixed(3)}.`);
  }
}
lines.push('', 'SELECTED TRAINING RECIPES: FIXED CONFIGURATIONS, SEEDS 11/17/23',
  'Seed 11 is reused with immutable checkpoint/history hashes; seeds 17 and 23 are new full-budget runs.');
for (const cohort of selectedConfirmations) {
  lines.push(`${cohort.configuration.operation}: ${cohort.complete ? 'all three complete' : 'confirmation pending'}; ${cohort.runs.length}/3 recorded.`);
  for (const run of cohort.runs) lines.push(`  Seed ${run.seed}: train/test ${percent(run.trainAccuracy)} / ${percent(run.testAccuracy)}, residue RMSE ${run.trainRmse.toFixed(3)} / ${run.testRmse.toFixed(3)}; checkpoint replay ${run.checkpointReplayed ? 'passed' : 'pending'}.`);
}
lines.push('', 'SMALL-TASK AND SUBSET CONTROLS (seed 11; best final training RMSE per dataset/scale)',
  'Control budgets use equal optimizer updates. Epoch counts and repeated visits per equation differ with dataset size.');
for (const [label, run] of controlGroups) lines.push(`${label}: train/test ${percent(run.finalTrainAccuracy)} / ${percent(run.finalTestAccuracy)}, RMSE ${run.finalTrainRmse.toFixed(3)} / ${run.finalTestRmse.toFixed(3)}; ${brief(run)}.`);
lines.push('', 'PAIRED FIRST-LAYER BIAS CONTROL (domain minus uniform; identical later biases)',
  'Negative train RMSE deltas favor domain anchoring. These are seed-11 screens, not three-seed conclusions.');
for (const effect of biasEffects) lines.push(`${effect.operation}/${effect.scale}: ${effect.finitePairs} finite lr pairs; median train RMSE delta ${effect.medianTrainRmseDomainMinusUniform?.toFixed(4) ?? 'pending'}.`);
lines.push('', 'DEEP PAIRED NORM/SKIP EFFECTS (fixed width, depth, optimizer, scale, other flag and seed 11)',
  'Enabled-minus-disabled deltas; short screens, not three-seed conclusions.');
for (const effect of deepEffects) lines.push(`${effect.operation}/${effect.scale}, ${effect.flag}: ${effect.finitePairs} finite pairs, ${effect.divergentPairs} divergent, ${effect.pendingPairs} pending; median train/test RMSE delta ${effect.medianTrainRmseEnabledMinusDisabled?.toFixed(3) ?? 'pending'} / ${effect.medianTestRmseEnabledMinusDisabled?.toFixed(3) ?? 'pending'}.`);
lines.push('', 'SINGLE-FACTOR COMPARISONS WITH THE FULL-BATCH W128 D8 PILOT',
  'Same seeded data and all unchanged settings. Batch comparisons keep epochs and visits per equation fixed; update counts differ.');
for (const row of optimizationAblations) lines.push(`${row.operation}, ${row.variant}: ${row.status}, ${row.epochs} epochs, ${row.updates} updates (reference ${row.referenceUpdates}); train/test ${percent(row.trainAccuracy)} / ${percent(row.testAccuracy)}, train RMSE delta ${row.trainRmseDelta?.toFixed(3) ?? 'divergence'}, held-out RMSE delta ${row.testRmseDelta?.toFixed(3) ?? 'divergence'}.`);
lines.push('', 'REPRESENTABILITY CONTROL: HAND-SET WEIGHTS, NO TRAINING');
for (const record of construction) lines.push(`${record.operation} mod 97: plain W${record.width} D${record.depth}, ${record.parameters} parameters; ${record.trainCorrect + record.heldoutCorrect}/${record.trainRows + record.heldoutRows} pairs exact in float32.`);
lines.push('This proves that the constructed architecture can express the complete rule; it does not demonstrate learning from Kaiming initialization.', '',
  'W128 RESIDUAL ABLATION (seeds 11/17/23; existing optimizer profiles)');
for (const profile of rotaryComparison) lines.push(`${profile.name}, lr=${profile.learningRate}, wd=${profile.weightDecay}:`,
  '  Epochs to held-out 99%, with skip: ' + profile.withResidualEpochs99.map(value => value ?? 'not reached').join(' / '),
  '  Epochs to held-out 99%, without skip: ' + profile.withoutResidualEpochs99.map(value => value ?? 'not reached').join(' / '),
  '  Without skip, final train/test: ' + profile.withoutResidualTrainAccuracy.map((value, index) => percent(value) + '/' + percent(profile.withoutResidualTestAccuracy[index])).join(' / '));
lines.push('The ablation changes only the skip, retaining both RMSNorm operations. It does not separately retune each no-skip optimizer.', '',
  'VALIDATION', `${runs.length} immutable recorded histories; ${savedPredictions} saved held-out predictions checked against modular targets and metrics.`,
  `${matrices.size} paired Kaiming matrix initializations; ${replayedTrials} identical repeated trials across phases.`,
  `${splitFingerprints.size} held-out splits checked for identical equations across paired configurations.`,
  `${prefixChecks.length} seed-11 confirmations checked against their actual shared screening rows.`,
  `${checkpointReplays.size} saved models reloaded on CUDA; exported train and held-out predictions reproduce the final logged metrics.`,
  'W128 paired histories and initial weights are also checked; its complete native evidence remains in rotary/summary.json.',
  'Scope: finite grids and compute budgets. Architecture, numeric representation and loss differ together from the tied-embedding classifier.',
  'No commits, pushes, deployments or uploads.');
lines.push('', 'PROTOCOL COVERAGE AND SELECTION AUDIT');
for (const coverage of gridCoverage) lines.push(coverage.trials ? `${coverage.name}: ${coverage.trials.recorded}/${coverage.trials.expected} planned trials.`
  : `${coverage.name}: ${coverage.screens.recorded}/${coverage.screens.expected} screens; ${coverage.finals.recorded}/${coverage.finals.expected ?? 'not yet selected'} final runs; ${coverage.selectionsChecked} selections independently checked.`);
writeFileSync(join(base, stem + '.txt'), lines.join('\n') + '\n');
const fields = ['phase', 'stage', 'variant', 'operation', 'modulus', 'scale', 'trainLimit', 'width', 'depth', 'preRmsnorm', 'residual', 'biasInitialization', 'biasScale', 'learningRate', 'weightDecay', 'beta2', 'batch', 'schedule', 'seed', 'epochs', 'updates', 'status', 'parameters', 'finalTrainRmse', 'finalTestRmse', 'finalTrainAccuracy', 'finalTestAccuracy', 'peakTrainAccuracy', 'peakTrainEpoch', 'peakTestAccuracy'];
writeFileSync(join(base, stem + '-runs.csv'), [fields.join(','), ...runs.map(run => fields.map(field => run[field] ?? '').join(','))].join('\n') + '\n');
console.log(`Wrote ${stem}: audited ${runs.length} native trials and ${savedPredictions} held-out predictions.`);
