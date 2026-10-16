// Diagnose scalar MLP fit using local Rust/CUDA; Node only orchestrates trials.
// Usage: node scripts/local-mlp-diagnostics.mjs <binary> <evidence-root> <controls|deep|followup> [--trials FILE --label NAME] [--dry-run]
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {readFileSync, writeFileSync, copyFileSync, mkdirSync, existsSync, renameSync, rmSync, openSync, closeSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {createHash} from 'node:crypto';

const root = resolve(new URL('..', import.meta.url).pathname);
const [binaryText, baseText, phase, ...flags] = process.argv.slice(2);
assert.ok(baseText && ['controls', 'deep', 'followup'].includes(phase), 'Provide the native binary, evidence root and controls|deep|followup.');
const option = name => flags.includes(name) ? flags[flags.indexOf(name) + 1] : undefined;
const label = option('--label') ?? phase;
assert.match(label, /^[a-z0-9-]+$/, 'Use a simple local output label.');
const binary = resolve(binaryText), base = resolve(baseText), output = join(base, 'mlp-' + label);
const planPath = join(root, 'experiments/local-mlp-diagnostics.json');
const plan = JSON.parse(readFileSync(planPath));
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const configKey = c => `${c.operation}-p${c.modulus}-${c.scale}-n${c.trainLimit ?? 'all'}-w${c.width}-d${c.depth}-norm${Number(c.preRmsnorm)}-skip${Number(c.residual)}-lr${c.learningRate}-wd${c.weightDecay}-${c.schedule}-floor${c.minLrRatio}-b2${c.beta2}-batch${c.batch}-bias${c.biasInitialization}-${c.biasScale}`;
const key = c => `${c.stage}-${configKey(c)}-e${c.epochs}-s${c.seed}`;
const trainRows = c => c.trainLimit ?? Math.floor(c.modulus * (c.modulus - Number(c.operation === 'div')) / 2);
const configurations = [];
const common = {beta2: plan.optimizer.beta2, minLrRatio: 0.001, biasInitialization: 'zero', biasScale: 1, seed: plan.screenSeed};
let trialsPath, followupPlan;
if (phase === 'controls') {
  const g = plan.controls;
  for (const dataset of g.datasets) for (const operation of g.operations) for (const scale of g.scales)
    for (const flags_ of g.flags) for (const learningRate of g.learningRates) {
      const c = {...common, ...dataset, operation, scale, ...flags_, learningRate, width: g.width, depth: g.depth,
        weightDecay: g.weightDecay, schedule: g.schedule, minLrRatio: g.minimumLrRatio, stage: 'control'};
      c.batch = Math.min(g.batch, trainRows(c));
      const stepsPerEpoch = Math.ceil(trainRows(c) / c.batch);
      assert.equal(g.updates % stepsPerEpoch, 0, 'Use a whole number of epochs at the fixed update budget.');
      c.epochs = g.updates / stepsPerEpoch;
      c.reportEvery = Math.max(1, Math.floor(c.epochs / 100));
      configurations.push(c);
    }
} else if (phase === 'deep') {
  const g = plan.deep;
  for (const width of g.widths) for (const depth of g.depths) for (const preRmsnorm of g.preRmsnorm)
    for (const residual of g.residual) for (const learningRate of g.learningRates) for (const weightDecay of g.weightDecays)
      for (const operation of g.operations) for (const scale of g.scales)
        configurations.push({...common, width, depth, preRmsnorm, residual, learningRate, weightDecay, operation, scale,
          modulus: g.modulus, trainLimit: null, schedule: g.schedule, batch: g.batch, epochs: g.screenEpochs,
          reportEvery: 10, stage: 'screen'});
} else {
  assert.ok(option('--trials') && option('--label'), 'Follow-ups need a concrete trials file and a distinct output label.');
  trialsPath = resolve(option('--trials'));
  followupPlan = JSON.parse(readFileSync(trialsPath));
  assert.ok(followupPlan.reason && Array.isArray(followupPlan.trials) && followupPlan.trials.length, 'Record a reason and explicit configurations before launch.');
  for (const trial of followupPlan.trials) {
    const config = {...common, modulus: 97, trainLimit: null, preRmsnorm: false, residual: false,
      weightDecay: 0, schedule: 'cosine', batch: 512, reportEvery: 10, ...trial, stage: 'followup'};
    for (const field of ['width', 'depth', 'epochs', 'batch', 'reportEvery', 'seed', 'modulus'])
      assert.ok(Number.isSafeInteger(config[field]) && config[field] > 0, 'Invalid ' + field);
    assert.ok(['div', 'mul'].includes(config.operation) && ['unit', 'raw', 'centered'].includes(config.scale));
    assert.ok(['zero', 'uniform', 'domain'].includes(config.biasInitialization));
    assert.ok(Number.isFinite(config.learningRate) && config.learningRate > 0);
    assert.ok(config.batch <= trainRows(config), 'Batch exceeds the training split.');
    configurations.push(config);
  }
}
assert.equal(new Set(configurations.map(key)).size, configurations.length);
if (flags.includes('--dry-run')) {
  console.log(JSON.stringify({phase, trials: configurations.length, maxFinalRuns: phase === 'deep' ? plan.deep.operations.length * plan.deep.scales.length * 2 * plan.seeds.length : 0,
    depths: [...new Set(configurations.map(c => c.depth))], widths: [...new Set(configurations.map(c => c.width))], localOnly: true}, null, 2));
  process.exit(0);
}
for (const component of ['rotary', 'scalar'])
  assert.equal(JSON.parse(readFileSync(join(base, component, 'source.json'))).completed, true, 'Complete the original experiment first.');
mkdirSync(output, {recursive: true});
const fingerprint = {phase, label, binarySha256: hash(binary), planSha256: hash(planPath), driverSha256: hash(new URL(import.meta.url)),
  ...(trialsPath ? {trialsSha256: hash(trialsPath)} : {}), localOnly: true};
if (existsSync(join(output, 'source.json'))) {
  const previous = JSON.parse(readFileSync(join(output, 'source.json')));
  for (const field of ['phase', 'label', 'binarySha256', 'planSha256', 'driverSha256', 'trialsSha256']) assert.equal(previous[field], fingerprint[field], 'Do not mix changed diagnostic inputs.');
}
const save = (name, value) => {
  writeFileSync(join(output, name + '.next'), JSON.stringify(value, null, 2) + '\n');
  renameSync(join(output, name + '.next'), join(output, name));
};
save('source.json', {...fingerprint, completed: false});
save('plan.json', plan);
copyFileSync(planPath, join(output, 'source-plan.json'));
copyFileSync(new URL(import.meta.url), join(output, 'source-driver.mjs'));
if (trialsPath) copyFileSync(trialsPath, join(output, 'source-trials.json'));
if (followupPlan) save('followup-plan.json', followupPlan);
let runs = existsSync(join(output, 'runs.json')) ? JSON.parse(readFileSync(join(output, 'runs.json'))) : [];
const children = new Set();
let stopped = false;
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => {
  stopped = true;
  for (const child of children) child.kill(signal);
});
function describe(directory, config) {
  const metadata = Object.fromEntries(readFileSync(join(directory, 'run.txt'), 'utf8').trim().split('\n').map(line => {
    const at = line.indexOf('='); return [line.slice(0, at), line.slice(at + 1)];
  }));
  const expected = {architecture: 'scalar-mlp', inputs: 2, outputs: 1, loss: 'mse_mean', operation: config.operation,
    modulus: config.modulus, scale: config.scale, width: config.width, depth: config.depth, pre_rmsnorm: config.preRmsnorm,
    residual: config.residual, training_equations: trainRows(config), seed: config.seed, requested_epochs: config.epochs,
    batch: config.batch, schedule: config.schedule, evaluation: 'complete_splits'};
  for (const [field, value] of Object.entries(expected)) assert.equal(metadata[field], String(value), `Mismatched ${field}: ${directory}`);
  for (const [field, value] of Object.entries({learning_rate: config.learningRate, weight_decay: config.weightDecay, minimum_lr_ratio: config.minLrRatio}))
    assert.ok(Math.abs(Number(metadata[field]) - value) < 1e-8, `Mismatched ${field}`);
  assert.equal(metadata.betas, `0.9,${config.beta2}`);
  assert.equal(metadata.bias_initialization ?? 'zero', config.biasInitialization);
  assert.ok(Math.abs(Number(metadata.bias_initialization_scale ?? 1) - config.biasScale) <= 1e-6 * Math.max(1, config.biasScale));
  assert.ok(['complete', 'nonfinite'].includes(metadata.status));
  const [header, ...lines] = readFileSync(join(directory, 'loss.csv'), 'utf8').trim().split('\n');
  const fields = header.split(',');
  const rows = lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [fields[index], Number(value)])));
  assert.ok(rows.length);
  for (const [index, row] of rows.entries()) {
    assert.ok(Object.values(row).every(Number.isFinite));
    assert.equal(row.epoch, Math.min(index * config.reportEvery, config.epochs));
    assert.equal(row.step, row.epoch * Math.ceil(trainRows(config) / config.batch));
  }
  if (metadata.status === 'complete') assert.equal(rows.at(-1).epoch, config.epochs);
  const final = rows.at(-1);
  return {status: metadata.status, parameters: Number(metadata.parameters), completedEpochs: Number(metadata.completed_epochs), updates: Number(metadata.updates),
    finalTrainMse: final.train_mse, finalTestMse: final.test_mse, finalTrainRmse: final.train_rmse_residue, finalTestRmse: final.test_rmse_residue,
    finalTrainAccuracy: final.train_accuracy, finalTestAccuracy: final.test_accuracy,
    bestTrainRmse: Math.min(...rows.map(row => row.train_rmse_residue)), bestTestRmse: Math.min(...rows.map(row => row.test_rmse_residue)),
    peakTrainAccuracy: Math.max(...rows.map(row => row.train_accuracy)), peakTestAccuracy: Math.max(...rows.map(row => row.test_accuracy)),
    constantTestRmse: Number(metadata.constant_test_rmse_residue), elapsedSeconds: Number(metadata.elapsed_seconds),
    historySha256: hash(join(directory, 'loss.csv')), initializationSha256: hash(join(directory, 'initial-model.bin'))};
}
async function execute(configs) {
  const queue = configs.filter(config => !runs.some(run => key(run) === key(config) && ['complete', 'nonfinite'].includes(run.status)));
  const workers = Array.from({length: 4}, async () => {
    for (let config; !stopped && (config = queue.shift());) {
      const directory = join(output, key(config));
      rmSync(directory, {recursive: true, force: true});
      mkdirSync(directory, {recursive: true});
      const args = ['--width', String(config.width), '--depth', String(config.depth), '--operation', config.operation, '--modulus', String(config.modulus),
        '--scale', config.scale, '--epochs', String(config.epochs), '--batch', String(config.batch), '--report-every', String(config.reportEvery),
        '--learning-rate', String(config.learningRate), '--weight-decay', String(config.weightDecay), '--beta2', String(config.beta2),
        '--schedule', config.schedule, '--min-lr-ratio', String(config.minLrRatio), '--seed', String(config.seed), '--output', directory,
        ...(config.biasInitialization === 'zero' && config.biasScale === 1 ? [] : ['--bias-init', config.biasInitialization, '--bias-scale', String(config.biasScale)]),
        ...(config.trainLimit === null ? [] : ['--train-limit', String(config.trainLimit)]),
        ...(config.preRmsnorm ? ['--pre-rmsnorm'] : []), ...(config.residual ? ['--residual'] : [])];
      writeFileSync(join(directory, 'arguments.json'), JSON.stringify({binary, args, config}, null, 2) + '\n');
      const log = openSync(join(directory, 'stdout.log'), 'w');
      const child = spawn(binary, args, {cwd: root, stdio: ['ignore', log, log]});
      children.add(child);
      let exitCode;
      try {exitCode = await new Promise((done, fail) => {child.once('error', fail); child.once('exit', done);});}
      finally {closeSync(log); children.delete(child);}
      if (stopped) return;
      assert.equal(exitCode, 0, `Native diagnostic failed: ${directory}`);
      const record = {...config, directory, exitCode, ...describe(directory, config)};
      runs = runs.filter(run => key(run) !== key(config)).concat(record);
      save('runs.json', runs);
      console.log(`${runs.length}: ${key(config)} ${record.status}; train ${(100 * record.finalTrainAccuracy).toFixed(2)}%, held-out ${(100 * record.finalTestAccuracy).toFixed(2)}%; train/test RMSE ${record.finalTrainRmse.toFixed(3)}/${record.finalTestRmse.toFixed(3)}`);
    }
  });
  try {await Promise.all(workers);}
  catch (error) {
    stopped = true;
    for (const child of children) child.kill('SIGTERM');
    throw error;
  }
  if (stopped) process.exit(130);
}
await execute(configurations);
assert.equal(runs.filter(run => run.stage === configurations[0].stage).length, configurations.length);
let selections = [], finalCount = 0;
if (phase === 'deep') {
  for (const operation of plan.deep.operations) for (const scale of plan.deep.scales) for (const criterion of ['Train', 'Test']) {
    const candidates = runs.filter(run => run.stage === 'screen' && run.status === 'complete' && run.operation === operation && run.scale === scale)
      .sort((a, b) => a[`final${criterion}Rmse`] - b[`final${criterion}Rmse`] || b[`final${criterion}Accuracy`] - a[`final${criterion}Accuracy`] || a.parameters - b.parameters);
    assert.ok(candidates.length);
    selections.push({operation, scale, criterion: criterion === 'Train' ? 'training-fit' : 'held-out-fit', screen: candidates[0]});
  }
  save('finalists.json', selections);
  const finals = [...new Map(selections.flatMap(({screen}) => plan.seeds.map(seed => ({...Object.fromEntries(Object.keys(configurations[0]).map(field => [field, screen[field]])),
    seed, stage: 'final', epochs: plan.deep.finalEpochs, reportEvery: 100}))).map(c => [key(c), c])).values()];
  finalCount = finals.length;
  await execute(finals);
  selections = selections.map(selection => ({...selection, runs: plan.seeds.map(seed => runs.find(run => run.stage === 'final' && run.seed === seed && configKey(run) === configKey(selection.screen)))}));
  assert.ok(selections.every(selection => selection.runs.every(Boolean)));
}
save('summary.json', {plan, phase, screens: configurations.length, finalRuns: finalCount, selections, runs: runs.length});
save('source.json', {...fingerprint, completed: true});
console.log(`COMPLETE: ${phase} diagnostics.`);
