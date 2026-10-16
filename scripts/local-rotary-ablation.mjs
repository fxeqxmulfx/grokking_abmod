// Paired, local-only CUDA skip ablation of the existing W128 website presets.
// Usage: node scripts/local-rotary-ablation.mjs <binary> <output>
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {readFileSync, writeFileSync, mkdirSync, existsSync, globSync, renameSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {createHash} from 'node:crypto';
import {ShortGrokking} from '../web/own-channel/dist/model.js';

const root = resolve(new URL('..', import.meta.url).pathname);
const [binaryText, outputText] = process.argv.slice(2);
if (!outputText) throw new Error('Provide a verified binary and local output directory.');
const binary = resolve(binaryText), output = resolve(outputText);
const plan = JSON.parse(readFileSync(join(root, 'experiments/local-residual-regression.json')));
const grid = plan.rotaryAblation;
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const metadata = directory => Object.fromEntries(readFileSync(join(directory, 'run.txt'), 'utf8').trim().split('\n').map(line => {const i = line.indexOf('='); return [line.slice(0, i), line.slice(i + 1)];}));
const key = run => `${run.profile}/${run.seed}/${run.residual}`;
mkdirSync(output, {recursive: true});
const save = (name, value) => {writeFileSync(join(output, name + '.next'), JSON.stringify(value, null, 2) + '\n'); renameSync(join(output, name + '.next'), join(output, name));};
let runs = existsSync(join(output, 'runs.json')) ? JSON.parse(readFileSync(join(output, 'runs.json'))) : [];
const caches = globSync(join(root, 'target/optimizer-grid/**/runs.json')).flatMap(path => JSON.parse(readFileSync(path)));
const children = new Set(); let stopped = false;
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => {stopped = true; for (const child of children) child.kill(signal);});
save('source.json', {completed: false, binarySha256: hash(binary)});

function history(directory) {
  const [header, ...lines] = readFileSync(join(directory, 'loss.csv'), 'utf8').trim().split('\n');
  const fields = header.split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [fields[index], Number(value)])));
}
function describe(directory) {
  const rows = history(directory), first = (name, limit) => rows.find(row => row[name] >= limit)?.completed_epochs ?? null;
  assert.ok(rows.every((row, epoch) => row.completed_epochs === epoch && row.step === epoch * 10 && Object.values(row).every(Number.isFinite)));
  const criterion = new ShortGrokking();
  for (const row of rows) criterion.observe({epoch: row.completed_epochs, trainAccuracy: row.train_accuracy, heldoutAccuracy: row.validation_accuracy});
  const firstTest99 = first('validation_accuracy', .99);
  assert.ok(firstTest99 !== null || rows.at(-1).completed_epochs === grid.epochs);
  return {completedEpochs: rows.at(-1).completed_epochs, firstTrain99: first('train_accuracy', .99), firstTest90: first('validation_accuracy', .9), firstTest99,
    peakTest: Math.max(...rows.map(row => row.validation_accuracy)), finalTest: rows.at(-1).validation_accuracy,
    finalTrain: rows.at(-1).train_accuracy, finalTestLoss: rows.at(-1).validation_cross_entropy,
    plateau: criterion.plateau, riseEpochs: criterion.rise, historySha256: hash(join(directory, 'loss.csv'))};
}
function reusable(config) {
  if (!config.residual) return undefined;
  return caches.find(run => {
    if (run.width !== 128 || run.seed !== config.seed || run.learningRate !== config.learningRate || run.weightDecay !== config.weightDecay || run.exitCode !== 0 || (run.firstTest99 === null && run.completedEpochs < grid.epochs)) return false;
    if (!existsSync(join(run.directory, 'loss.csv'))) return false;
    const meta = metadata(run.directory);
    return meta.architecture === grid.architecture && meta.depth === '1' && meta.operation === 'div' && meta.dense_embedding_initialization === 'normal-width' && meta.batch === '512' && meta.betas === '0.9,0.98' && meta.dense_residual !== 'false';
  });
}
const queue = grid.profiles.flatMap(profile => plan.seeds.flatMap(seed => grid.residual.map(residual => ({profile: profile.name, seed, residual, learningRate: profile.learningRate, weightDecay: profile.weightDecay}))));
await Promise.all(Array.from({length: 4}, async () => {
  for (let config; !stopped && (config = queue.shift());) {
    if (runs.some(run => key(run) === key(config) && run.exitCode === 0)) continue;
    const reused = reusable(config);
    let directory, exitCode = 0, reusedFrom = null;
    if (reused) {directory = reused.directory; reusedFrom = directory;}
    else {
      directory = join(output, `${config.profile}-s${config.seed}-res${Number(config.residual)}`);
      mkdirSync(directory, {recursive: true});
      const args = ['grok-simple', '--model', grid.architecture, '--width', '128', '--depth', '1', '--activation', 'relu', '--encoding', 'one-hot', '--dense-embedding-initialization', 'normal-width', '--operation', 'div', '--modulus', '97', '--train-fraction', '0.5', '--learning-rate', String(config.learningRate), '--weight-decay', String(config.weightDecay), '--seed', String(config.seed), '--batch', '512', '--epochs', String(grid.epochs), '--report-every', '10', '--stop-validation-accuracy', '.99', '--output', directory, ...(!config.residual ? ['--no-residual'] : [])];
      writeFileSync(join(directory, 'arguments.json'), JSON.stringify({binary, args}, null, 2) + '\n');
      const child = spawn(binary, args, {cwd: root, stdio: ['ignore', 'ignore', 'inherit']}); children.add(child);
      exitCode = await new Promise((done, fail) => {child.once('error', fail); child.once('exit', done);}); children.delete(child);
      if (stopped) return;
      assert.equal(exitCode, 0, 'CUDA ablation failed: ' + directory);
    }
    const record = {...config, width: 128, directory, exitCode, reusedFrom, ...describe(directory)};
    runs.push(record); save('runs.json', runs);
    console.log(`${key(config)}: test99=${record.firstTest99 ?? '-'}, peak=${(record.peakTest * 100).toFixed(2)}%, plateau=${record.plateau.epochs}${reused ? ' (reused control)' : ''}`);
  }
}));
if (stopped) process.exit(130);
assert.equal(runs.length, 18);
save('summary.json', {plan: grid, seeds: plan.seeds, runs});
save('source.json', {completed: true, binarySha256: hash(binary), driverSha256: hash(new URL(import.meta.url)), planSha256: hash(join(root, 'experiments/local-residual-regression.json')), localOnly: true});
console.log('COMPLETE: 9 paired W128 comparisons.');
