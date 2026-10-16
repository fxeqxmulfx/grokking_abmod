// Local orchestration only; every model, gradient and optimizer step uses Rust/CUDA.
// Usage: node scripts/local-scalar-grid.mjs <binary> <output> [--dry-run]
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {readFileSync, writeFileSync, existsSync, mkdirSync, renameSync, rmSync, openSync, closeSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {createHash} from 'node:crypto';

const root = resolve(new URL('..', import.meta.url).pathname);
const [binaryText, outputText, ...flags] = process.argv.slice(2);
if (!outputText) throw new Error('Provide the scalar-mlp CUDA binary and a local output directory.');
const binary = resolve(binaryText), output = resolve(outputText);
const planPath = join(root, 'experiments/local-residual-regression.json');
const plan = JSON.parse(readFileSync(planPath)), grid = plan.scalarMlp;
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const configKey = c => `${c.operation}-${c.scale}-w${c.width}-d${c.depth}-n${Number(c.preRmsnorm)}-r${Number(c.residual)}-lr${c.learningRate}-wd${c.weightDecay}`;
const key = c => `${c.stage}-${configKey(c)}-s${c.seed}`;
const families = grid.operations.flatMap(operation => grid.scales.flatMap(scale => grid.preRmsnorm.flatMap(preRmsnorm => grid.residual.map(residual => ({operation, scale, preRmsnorm, residual})))));
const familyKey = c => `${c.operation}/${c.scale}/${Number(c.preRmsnorm)}/${Number(c.residual)}`;
const belongs = (run, family) => run.operation === family.operation && run.scale === family.scale && (run.depth === 1 || run.preRmsnorm === family.preRmsnorm && run.residual === family.residual);
const screens = [];
for (const width of grid.widths) for (const depth of grid.depths)
  for (const preRmsnorm of depth === 1 ? [false] : grid.preRmsnorm)
    for (const residual of depth === 1 ? [false] : grid.residual)
      for (const learningRate of grid.learningRates) for (const weightDecay of grid.weightDecays)
        for (const operation of grid.operations) for (const scale of grid.scales)
          screens.push({width, depth, preRmsnorm, residual, learningRate, weightDecay, operation, scale, seed: grid.screenSeed, stage: 'screen', epochs: grid.screenEpochs});
assert.equal(new Set(screens.map(key)).size, screens.length);
if (flags.includes('--dry-run')) {
  console.log(JSON.stringify({screens: screens.length, families: families.length, maximumFinalRuns: families.length * plan.seeds.length,
    screenEpochs: grid.screenEpochs, finalEpochs: grid.finalEpochs}, null, 2)); process.exit(0);
}
mkdirSync(output, {recursive: true});
const save = (name, value) => {writeFileSync(join(output, name + '.next'), JSON.stringify(value, null, 2) + '\n'); renameSync(join(output, name + '.next'), join(output, name));};
const fingerprint = {binarySha256: hash(binary), planSha256: hash(planPath), driverSha256: hash(new URL(import.meta.url)), localOnly: true};
if (existsSync(join(output, 'source.json'))) {
  const previous = JSON.parse(readFileSync(join(output, 'source.json')));
  assert.equal(previous.binarySha256, fingerprint.binarySha256, 'Cannot mix binaries within one grid.');
  assert.equal(previous.planSha256, fingerprint.planSha256, 'Cannot mix changed experiment settings.');
}
save('source.json', {...fingerprint, completed: false});
save('plan.json', plan);
let runs = existsSync(join(output, 'runs.json')) ? JSON.parse(readFileSync(join(output, 'runs.json'))) : [];
const children = new Set(); let stopped = false;
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => {stopped = true; for (const child of children) child.kill(signal);});
function describe(directory, config) {
  const metadata = Object.fromEntries(readFileSync(join(directory, 'run.txt'), 'utf8').trim().split('\n').map(line => {const at = line.indexOf('='); return [line.slice(0, at), line.slice(at + 1)];}));
  for (const [field, value] of Object.entries({architecture: 'scalar-mlp', inputs: 2, outputs: 1, loss: 'mse_mean', operation: config.operation,
    scale: config.scale, width: config.width, depth: config.depth, pre_rmsnorm: config.preRmsnorm, residual: config.residual,
    seed: config.seed, requested_epochs: config.epochs, batch: plan.batch, evaluation: 'complete_splits'})) assert.equal(metadata[field], String(value), `Mismatched ${field}: ${directory}`);
  assert.ok(['complete', 'nonfinite'].includes(metadata.status));
  const [header, ...lines] = readFileSync(join(directory, 'loss.csv'), 'utf8').trim().split('\n'), fields = header.split(',');
  const rows = lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [fields[index], Number(value)])));
  assert.ok(rows.length > 0);
  const interval = Number(metadata.report_every_epochs);
  for (const [index, row] of rows.entries()) {
    assert.ok(Object.values(row).every(Number.isFinite), 'Nonfinite recorded metrics.');
    assert.equal(row.epoch, index * interval);
    assert.equal(row.step, row.epoch * Math.ceil(Number(metadata.training_equations) / plan.batch));
  }
  if (metadata.status === 'complete') {
    assert.equal(rows.at(-1).epoch, config.epochs);
    assert.ok(existsSync(join(directory, 'model.bin')));
  }
  const final = rows.at(-1);
  return {status: metadata.status, parameters: Number(metadata.parameters), completedEpochs: Number(metadata.completed_epochs),
    finalTrainMse: final.train_mse, finalTestMse: final.test_mse, finalTrainRmse: final.train_rmse_residue, finalTestRmse: final.test_rmse_residue,
    finalTrainAccuracy: final.train_accuracy, finalTestAccuracy: final.test_accuracy,
    bestTestRmse: Math.min(...rows.map(row => row.test_rmse_residue)), peakTestAccuracy: Math.max(...rows.map(row => row.test_accuracy)),
    constantTestRmse: Number(metadata.constant_test_rmse_residue), constantTestAccuracy: Number(metadata.constant_test_accuracy),
    elapsedSeconds: Number(metadata.elapsed_seconds), historySha256: hash(join(directory, 'loss.csv'))};
}
async function execute(configurations) {
  const queue = configurations.filter(config => !runs.some(run => key(run) === key(config) && ['complete', 'nonfinite'].includes(run.status)));
  await Promise.all(Array.from({length: 4}, async () => {
    for (let config; !stopped && (config = queue.shift());) {
      const directory = join(output, key(config));
      rmSync(directory, {recursive: true, force: true}); mkdirSync(directory, {recursive: true});
      const args = ['--width', String(config.width), '--depth', String(config.depth), '--operation', config.operation, '--scale', config.scale,
        '--epochs', String(config.epochs), '--batch', String(plan.batch), '--report-every', String(config.stage === 'screen' ? grid.screenReportEveryEpochs : grid.finalReportEveryEpochs),
        '--learning-rate', String(config.learningRate), '--weight-decay', String(config.weightDecay), '--seed', String(config.seed), '--output', directory,
        ...(config.preRmsnorm ? ['--pre-rmsnorm'] : []), ...(config.residual ? ['--residual'] : [])];
      writeFileSync(join(directory, 'arguments.json'), JSON.stringify({binary, args, config}, null, 2) + '\n');
      const log = openSync(join(directory, 'stdout.log'), 'w');
      const child = spawn(binary, args, {cwd: root, stdio: ['ignore', log, log]}); children.add(child);
      const exitCode = await new Promise((done, fail) => {child.once('error', fail); child.once('exit', done);});
      closeSync(log); children.delete(child);
      if (stopped) return;
      assert.equal(exitCode, 0, 'CUDA scalar trial failed: ' + directory);
      const record = {...config, directory, exitCode, ...describe(directory, config)};
      runs = runs.filter(run => key(run) !== key(config)).concat(record); save('runs.json', runs);
      console.log(`${runs.length}: ${key(config)}: ${record.status}, test RMSE ${record.finalTestRmse.toFixed(3)}, rounded accuracy ${(100 * record.finalTestAccuracy).toFixed(2)}%, ${record.elapsedSeconds.toFixed(1)} s`);
    }
  }));
  if (stopped) process.exit(130);
}
await execute(screens);
assert.equal(runs.filter(run => run.stage === 'screen').length, screens.length);
const rank = (a, b) => a.finalTestRmse - b.finalTestRmse || b.finalTestAccuracy - a.finalTestAccuracy || a.parameters - b.parameters;
const selected = families.map(family => {
  const candidates = runs.filter(run => run.stage === 'screen' && run.status === 'complete' && belongs(run, family)).sort(rank);
  assert.ok(candidates.length, 'No finite screen for family ' + familyKey(family));
  return {family, configuration: candidates[0]};
});
save('finalists.json', selected);
const finals = [...new Map(selected.flatMap(({configuration}) => plan.seeds.map(seed => ({...Object.fromEntries(Object.entries(configuration).filter(([field]) => ['width', 'depth', 'preRmsnorm', 'residual', 'learningRate', 'weightDecay', 'operation', 'scale'].includes(field))), seed, stage: 'final', epochs: grid.finalEpochs}))).map(config => [key(config), config])).values()];
await execute(finals);
const results = selected.map(({family, configuration}) => ({family, screen: configuration,
  runs: plan.seeds.map(seed => runs.find(run => run.stage === 'final' && run.seed === seed && configKey(run) === configKey(configuration)))}));
assert.ok(results.every(result => result.runs.every(Boolean)));
save('summary.json', {plan, screenConfigurations: screens.length, finalConfigurations: finals.length, runs: runs.length, results});
save('source.json', {...fingerprint, completed: true});
console.log('COMPLETE: all scalar screens and selected three-seed comparisons.');
