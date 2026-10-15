import assert from 'node:assert/strict';
import {CDP} from './cdp.ts';

interface Work {frames: number; clears: Record<string, number>; rasters: number; snapshots: number[]}
const client = await CDP.connect(Number(process.env.GROKKING_CDP_PORT || 9224));
const origin = process.env.GROKKING_PREVIEW_ORIGIN || 'http://127.0.0.1:8080';
const evaluate = client.evaluate.bind(client), wait = client.wait.bind(client);
const paint = () => evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
const reset = () => evaluate('window.renderReview.reset()');
const work = () => evaluate<Work>('window.renderReview.work');

try {
  await client.call('Runtime.enable'); await client.call('Page.enable');
  await client.call('Emulation.setDeviceMetricsOverride', {width: 1366, height: 768, deviceScaleFactor: 1, mobile: false});
  // Instrument browser APIs, independently of the application's render caches.
  await client.call('Page.addScriptToEvaluateOnNewDocument', {source: `(() => {
    window.renderReview = {reset() {this.work = {frames: 0, clears: {}, rasters: 0, snapshots: []}}};
    window.renderReview.bitmaps = [];
    window.renderReview.reset();
    const raf = window.requestAnimationFrame;
    window.requestAnimationFrame = function(callback) {
      return raf.call(this, time => {window.renderReview.work.frames++; callback(time)});
    };
    const clear = CanvasRenderingContext2D.prototype.clearRect;
    CanvasRenderingContext2D.prototype.clearRect = function(...args) {
      const id = this.canvas.id, counts = window.renderReview.work.clears;
      if (id) counts[id] = (counts[id] || 0) + 1;
      return clear.apply(this, args);
    };
    const put = CanvasRenderingContext2D.prototype.putImageData;
    CanvasRenderingContext2D.prototype.putImageData = function(...args) {
      if (!this.canvas.id) {
        window.renderReview.work.rasters++;
        if (!window.renderReview.bitmaps.includes(this.canvas)) window.renderReview.bitmaps.push(this.canvas);
      }
      return put.apply(this, args);
    };
    const NativeWorker = Worker;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', ({data}) => {
          if (data.type === 'weights') window.renderReview.work.snapshots.push(data.step);
        });
      }
    };
  })()`});
  await client.call('Page.navigate', {url: origin + '/?lang=en'});
  await wait(`window.renderReview && document.querySelector('#task-dialog')?.open && document.querySelector('#status').textContent === 'Ready'`);
  await evaluate(`document.querySelector('#close-task').click(); document.querySelector('[data-view-tab="weights"]').click()`);
  await evaluate('document.fonts.ready');
  await evaluate('new Promise(resolve => setTimeout(resolve, 150))');
  await reset();
  await evaluate('new Promise(resolve => setTimeout(resolve, 250))');
  assert.deepEqual(await work(), {frames: 0, clears: {}, rasters: 0, snapshots: []}, 'Idle must schedule no frames or redraws');

  await evaluate(`(async () => {
    const canvas = document.querySelector('#input-b'), rect = canvas.getBoundingClientRect();
    for (let i = 0; i < 12; i++) {
      canvas.dispatchEvent(new MouseEvent('mousemove', {clientX: rect.left + 10 + 3 * i, clientY: rect.top + 30}));
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
  })()`);
  assert.deepEqual((await work()).clears, {}, 'Changing a hover tooltip must not redraw the atlas');
  assert.equal((await work()).rasters, 0);
  assert.match(await evaluate<string>(`document.querySelector('#input-b').title`), /^A\[/);

  await reset();
  await evaluate(`(() => {const input = document.querySelector('#channel-select'); input.value = '32'; input.dispatchEvent(new Event('input'))})()`);
  await paint();
  const selection = await work();
  assert.deepEqual(selection.clears, {'input-a': 1, 'input-b': 1, readout: 1});
  assert.equal(selection.rasters, 0, 'Selection must reuse all three heatmap rasters');

  await reset();
  await evaluate(`(() => {
    const canvas=document.getElementById('input-a'),context=canvas.getContext('2d');
    window.renderReview.pixels=context.getImageData(0,0,canvas.width,canvas.height).data;
    context.reset();
    const bitmap=window.renderReview.bitmaps[0];bitmap.getContext('2d').reset();bitmap.dispatchEvent(new Event('contextrestored'));
  })()`);
  await paint();
  assert.equal(await evaluate(`(() => {const c=document.getElementById('input-a'),pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;return pixels.every((value,i)=>value===window.renderReview.pixels[i])})()`), true, 'A restored source bitmap must regenerate the original weight colors.');
  assert.equal((await work()).rasters, 1, 'Restore only the discarded source bitmap.');
  assert.deepEqual((await work()).clears, {'input-a': 1});
  await evaluate('delete window.renderReview.pixels');

  // Receive real per-epoch metrics more often than the one-Hz weight snapshots.
  await evaluate(`(() => {
    const capture = document.querySelector('#capture'); capture.value = '1'; capture.dispatchEvent(new Event('change'));
    const limit = document.querySelector('#limit'); limit.value = '20'; limit.dispatchEvent(new Event('change'));
  })()`);
  await reset();
  await evaluate(`document.querySelector('#train').click()`);
  await wait(`document.querySelector('#status').textContent === 'Epoch limit reached' && document.querySelector('#epoch').textContent === '20'`);
  await paint();
  const training = await work();
  assert.equal(await evaluate(`document.querySelector('#observations').textContent`), '21 observations', 'Retain every evaluated epoch');
  assert.equal(training.snapshots.at(-1), 200, 'The final trained weights must arrive');
  assert.equal(new Set(training.snapshots).size, training.snapshots.length, 'Do not resend an unchanged weight snapshot');
  for (const id of ['input-a', 'input-b', 'readout']) assert.ok((training.clears[id] ?? 0) <= training.snapshots.length, 'Metrics must not repaint unchanged weights');
  assert.ok(training.rasters <= 3 * training.snapshots.length);
  await reset();
  await evaluate(`document.querySelector('#step').click()`);
  await evaluate('new Promise(resolve => setTimeout(resolve, 200))');
  assert.deepEqual((await work()).snapshots, [], 'An action at the epoch limit must not resend the same weights');
  assert.deepEqual((await work()).clears, {}, 'Status alone must not repaint the atlas');

  // The same metadata isolation applies to the complete network and the plot.
  await evaluate(`document.querySelector('[data-view-tab="channels"]').click()`); await paint();
  const helper = await evaluate(`document.querySelector('#mesh-helper').textContent`);
  await reset();
  await evaluate(`document.querySelector('#step').click()`);
  await evaluate('new Promise(resolve => setTimeout(resolve, 200))');
  assert.deepEqual((await work()).clears, {});
  assert.equal(await evaluate(`document.querySelector('#mesh-helper').textContent`), helper, 'Status must preserve the current network caption');
  await evaluate(`document.querySelector('[data-view-tab="training"]').click()`); await paint();
  await reset();
  await evaluate(`document.querySelector('#step').click()`);
  await evaluate('new Promise(resolve => setTimeout(resolve, 200))');
  assert.deepEqual((await work()).clears, {}, 'Status without a new observation must preserve the plot');

  assert.deepEqual(client.errors, []);
  console.log(JSON.stringify({selection, training}));
  console.log('PASS: idle, hover, selection, source bitmap restoration, snapshot cadence, complete epoch history, unchanged-weight deduplication and status-only updates.');
} finally {
  client.close();
}
