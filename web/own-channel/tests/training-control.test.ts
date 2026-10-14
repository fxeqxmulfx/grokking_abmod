import test from 'node:test';
import assert from 'node:assert/strict';
import {AccuracyPause} from '../dist/training-control.js';

test('99% test crossing pauses once, and continuing through a dip keeps the acknowledgement', () => {
  const target = new AccuracyPause();
  for (const accuracy of [0, .98, .98999, NaN]) assert.equal(target.observe(accuracy), false);
  assert.equal(target.observe(.99), true);
  for (const accuracy of [1, .99, .9, .99, 1]) assert.equal(target.observe(accuracy), false);
  assert.equal(new AccuracyPause().observe(.99), true, 'A reset re-arms the target.');
});
