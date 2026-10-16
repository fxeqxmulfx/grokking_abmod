// Validate complete CUDA histories, including a search stopped by the user.
// Usage: node scripts/analyze-width-floor.mjs <output> [plan]
import assert from 'node:assert/strict';
import {readFile, writeFile, rename} from 'node:fs/promises';
import {resolve, join} from 'node:path';
import {createHash} from 'node:crypto';

const directory = resolve(process.argv[2] ?? 'target/optimizer-grid/below-w16-2026-10-15');
const planPath = resolve(process.argv[3] ?? 'experiments/div97-dense-below16-grid.json');
const plan = JSON.parse(await readFile(planPath, 'utf8'));
const provenance = JSON.parse(await readFile(join(directory, 'source.json'), 'utf8'));
assert.equal(provenance.completed, true, 'Finish collecting the retained CUDA trials before analyzing.');
const runs = JSON.parse(await readFile(join(directory, 'runs.json'), 'utf8'));
const key = run => `${run.width}/${run.seed}/${run.learningRate}/${run.weightDecay}`;
assert.equal(new Set(runs.map(key)).size, runs.length, 'Duplicate trial configurations.');
const seeds = [plan.screenSeed, ...plan.confirmationSeeds];
const hash = data => createHash('sha256').update(data).digest('hex');
const save = async (name, object) => {
  const path = join(directory, name);
  await writeFile(path + '.next', JSON.stringify(object, null, 2) + '\n');
  await rename(path + '.next', path);
};
for (const run of runs) {
  assert.equal(run.exitCode, 0, `Failed trial: ${key(run)}`);
  assert.ok(!run.error, `Invalid trial: ${key(run)}`);
  const metadata = Object.fromEntries((await readFile(join(run.directory, 'run.txt'), 'utf8')).trim().split('\n').map(line => {
    const at = line.indexOf('='); return [line.slice(0, at), line.slice(at + 1)];
  }));
  for (const [field, value] of Object.entries({architecture: plan.architecture, activation: plan.fixed.activation,
    encoding: plan.fixed.encoding, operation: plan.fixed.operation, dense_embedding_initialization: plan.fixed.embeddingInitialization,
    depth: plan.fixed.depth, modulus: plan.fixed.modulus, train_fraction: plan.fixed.trainFraction, batch: plan.fixed.batch,
    width: run.width, seed: run.seed, parameters: 97 * run.width + run.width * run.width,
    learning_rate: run.learningRate, weight_decay: run.weightDecay, warmup_steps: plan.fixed.warmupUpdates,
    training_equations: 4656, validation_equations: 4656, evaluation: 'complete_splits',
    bias: false, normalization: 'pre_rmsnorm_blocks_and_final', betas: `${plan.fixed.beta1},${plan.fixed.beta2}`}))
    assert.equal(metadata[field], String(value), `Mismatched ${field}: ${key(run)}`);
  const csv = await readFile(join(run.directory, 'loss.csv'), 'utf8');
  const [header, ...lines] = csv.trim().split('\n'), fields = header.split(',');
  const rows = lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [fields[index], Number(value)])));
  for (const [index, row] of rows.entries()) {
    assert.ok(Object.values(row).every(Number.isFinite), `Nonfinite metrics: ${key(run)} epoch ${index}`);
    assert.equal(row.completed_epochs, index, `Missing epoch: ${key(run)}`);
    assert.equal(row.step, index * 10, `Incomplete epoch: ${key(run)}`);
    assert.equal(row.training_examples_seen, index * 4656, `Incorrect training split: ${key(run)}`);
  }
  const first = (field, threshold) => rows.find(row => row[field] >= threshold)?.completed_epochs ?? null;
  assert.equal(first('validation_accuracy', plan.stopTestAccuracy), run.firstTest99, `Incorrect crossing: ${key(run)}`);
  assert.equal(Math.max(...rows.map(row => row.validation_accuracy)), run.peakTest, `Incorrect peak accuracy: ${key(run)}`);
  assert.equal(rows.at(-1).completed_epochs, run.completedEpochs);
  assert.ok(run.firstTest99 !== null || run.completedEpochs >= plan.epochs, `Censored before the full budget: ${key(run)}`);
  Object.assign(run, {
    firstTrain99: first('train_accuracy', .99), firstTest90: first('validation_accuracy', .9),
    finalTrain: rows.at(-1).train_accuracy, finalTest: rows.at(-1).validation_accuracy,
    finalTestLoss: rows.at(-1).validation_cross_entropy,
    historySha256: hash(csv),
  });
}
const configurations = new Map();
for (const grid of [plan, ...(plan.refinements ?? [])]) for (const width of grid.widths)
  for (const learningRate of grid.learningRates) for (const weightDecay of grid.weightDecays)
    configurations.set(`${width}/${learningRate}/${weightDecay}`, {width, learningRate, weightDecay});
for (const run of runs) assert.ok(configurations.has(`${run.width}/${run.learningRate}/${run.weightDecay}`) && seeds.includes(run.seed), `Unplanned trial: ${key(run)}`);
const profiles = [...configurations.values()].map(config => {
  const selected = seeds.map(seed => runs.find(run => key(run) === key({...config, seed})));
  assert.ok(selected[0], `Unsearched configuration: ${JSON.stringify(config)}`);
  if (selected[0].firstTest99 !== null && (plan.confirmationWidths ?? plan.widths).includes(config.width))
    assert.ok(selected.every(Boolean), `Missing confirmation: ${JSON.stringify(config)}`);
  const epochs = selected.map(run => run?.firstTest99 ?? plan.epochs + 1).sort((a, b) => a - b);
  const successes = selected.filter(run => run?.firstTest99 != null).length;
  return {...config, successes, completeSeeds: selected.filter(Boolean).length,
    outcome: successes === seeds.length ? 'confirmed' : selected.some(run => run && run.firstTest99 === null) ? 'not-confirmed' : 'incomplete',
    maximumEpoch: Math.max(...epochs), medianEpoch: epochs[Math.floor(epochs.length / 2)],
    runs: selected.filter(Boolean)};
});
const rank = (a, b) => b.successes - a.successes || a.maximumEpoch - b.maximumEpoch || a.medianEpoch - b.medianEpoch || b.runs[0].peakTest - a.runs[0].peakTest;
const results = [...new Set(profiles.map(profile => profile.width))].sort((a, b) => a - b).map(width => {
  const own = profiles.filter(profile => profile.width === width).sort(rank);
  return {width, parameters: 97 * width + width * width,
    screenedConfigurations: own.length, successfulScreens: own.filter(profile => profile.runs[0].firstTest99 !== null).length,
    missingPositiveConfirmations: own.filter(profile => profile.runs[0].firstTest99 !== null).reduce((total, profile) => total + seeds.length - profile.completeSeeds, 0),
    peakScreenTestAccuracy: Math.max(...own.map(profile => profile.runs[0].peakTest)),
    winner: own.find(profile => profile.successes === seeds.length) ?? null, profiles: own};
});
const minimum = results.find(result => result.winner)?.width ?? null;
const minimumScreen = results.find(result => result.successfulScreens > 0)?.width ?? null;
await save('runs.json', runs);
await save('summary.json', {plan, seeds, runs: runs.length, minimumScreenWidth: minimumScreen, minimumRobustWidth: minimum,
  scope: 'Smallest width confirmed in the retained finite search; incomplete W14 confirmations and the epoch budget prevent a mathematical lower-bound claim.', results});
await save('source.json', {...provenance, planSha256: hash(await readFile(planPath)), analyzerSha256: hash(await readFile(new URL(import.meta.url)))});
console.log(`Validated ${runs.length} complete learning curves; smallest width confirmed at 99% on every seed: ${minimum ?? 'none'}.`);
for (const result of results) console.log(`W${result.width}: ${result.successfulScreens}/${result.screenedConfigurations} successful screens, peak ${(100 * result.peakScreenTestAccuracy).toFixed(2)}%, ` +
  (result.winner ? `lr ${result.winner.learningRate}, wd ${result.winner.weightDecay}, epochs ${result.winner.runs.map(run => run.firstTest99).join('/')}` : 'no configuration confirmed on every seed') +
  (result.missingPositiveConfirmations ? `; ${result.missingPositiveConfirmations} confirmations stopped by the user` : ''));
