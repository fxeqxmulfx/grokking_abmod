// Replay the actual browser classifier against every completed trajectory.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {ShortGrokking} from '../web/own-channel/dist/model.js';

const root = process.argv[2] ?? new URL('../target/rotate90-swiglu/', import.meta.url).pathname;
const rows = JSON.parse(fs.readFileSync(`${root}/full-results.json`, 'utf8'));
let verified = 0;
for (const result of rows) {
  for (const mode of ['evaluation', 'query']) {
    if (result[mode] === null) continue;
    const filename = mode === 'query' ? 'query-history.json' : 'history.json';
    const history = JSON.parse(fs.readFileSync(`${root}/runs/${result.config.id}/${filename}`, 'utf8'));
    const criterion = new ShortGrokking();
    for (const row of history) criterion.observe({epoch:row.epoch, trainAccuracy:row.train_accuracy, heldoutAccuracy:row.heldout_accuracy});
    const expected = result[mode].criterion;
    for (const key of ['plateau', 'perfect', 'rise', 'first90']) assert.deepEqual(criterion[key], expected[key]);
    assert.equal(criterion.event?.delayed ?? null, expected.event?.delayed ?? null);
    assert.equal(criterion.event?.epoch ?? null, expected.event?.epoch ?? null);
    verified++;
  }
}
const result = {
  browser_criterion_trajectories_verified:verified,
  browser_model_sha256:crypto.createHash('sha256').update(fs.readFileSync(new URL('../web/own-channel/dist/model.js', import.meta.url))).digest('hex'),
};
fs.writeFileSync(`${root}/criterion-verification.json`, JSON.stringify(result, null, 2)+'\n');
console.log(JSON.stringify(result));
