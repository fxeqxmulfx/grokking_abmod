import {mkdir, readFile, unlink, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import assert from 'node:assert/strict';
import {CDP} from './cdp.ts';
import {MODEL_IDS} from '../dist/types.js';
import {modelInfo, modelPresets} from '../dist/models.js';
import {pytorchSource} from '../dist/pytorch-export.js';

const origin = process.env.GROKKING_PREVIEW_ORIGIN ?? 'http://127.0.0.1:8080';
const directory = '/tmp/grokking-pytorch-downloads';
await mkdir(directory, {recursive: true});
const client = await CDP.connect(Number(process.env.GROKKING_CDP_PORT ?? 9224));
const evaluate = client.evaluate.bind(client), wait = client.wait.bind(client);
const row = `({epoch:Number(document.getElementById('epoch').textContent.replaceAll(',','')), accuracy:parseFloat(document.getElementById('heldout-accuracy').textContent), observations:document.getElementById('observations').textContent, workers:window.__labWorkers.length})`;
const records: unknown[] = [];
async function download(name: string) {
  const path = join(directory, name);
  await unlink(path).catch(error => {if (error.code !== 'ENOENT') throw error;});
  await evaluate(`document.getElementById('export-pytorch').click()`);
  const start = Date.now();
  while (Date.now() - start < 10000) {
    try {return await readFile(path, 'utf8');} catch {await new Promise(resolve => setTimeout(resolve, 100));}
  }
  throw new Error(`Missing PyTorch download: ${path}`);
}
try {
  await client.call('Runtime.enable'); await client.call('Page.enable');
  const injection = await client.call<{identifier: string}>('Page.addScriptToEvaluateOnNewDocument', {source: `window.__labWorkers=[];const NativeWorker=Worker;window.Worker=class extends NativeWorker {constructor(...args){super(...args);window.__labWorkers.push(this)}};localStorage.setItem('grokking.locale','en');`});
  await client.call('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: directory});
  await client.call('Emulation.setDeviceMetricsOverride', {width: 1366, height: 768, deviceScaleFactor: 1, mobile: false});
  await client.call('Page.navigate', {url: origin});
  await wait(`document.documentElement?.lang==='en'&&document.getElementById('task-dialog')?.open`);
  await evaluate(`document.getElementById('task-open').click()`);
  await wait(`document.getElementById('status').textContent==='Ready'`);
  assert.equal(await evaluate(`document.querySelectorAll('[data-model-tab]').length`), MODEL_IDS.length);
  assert.ok(await evaluate(`!document.getElementById('export-weights')&&!document.getElementById('stop-target')&&!!document.querySelector('a[href="https://arxiv.org/abs/2201.02177"]')`));
  for (const model of MODEL_IDS) for (const preset of modelPresets(model)) {
    const {width, parameters} = modelInfo(model), name = `rotate90-w${width}${preset === 'fast' ? '' : '-' + preset}-seed11.py`;
    await evaluate(`document.querySelector('[data-model-tab="${model}"]').click()`);
    if (modelPresets(model).length > 1) await evaluate(`document.querySelector('[data-preset-tab="${preset}"]').click()`);
    await wait(`document.getElementById('status').textContent==='Ready'&&document.getElementById('model-detail').textContent.includes('${width} channels')&&document.querySelector('[data-preset-tab="${preset}"]').getAttribute('aria-pressed')==='true'`);
    await evaluate(`document.getElementById('limit').value='3';document.getElementById('limit').dispatchEvent(new Event('change'))`);
    const source = await download(name);
    assert.equal(source, pytorchSource({model, preset, seed: 11, maxEpochs: 3}));
    await evaluate(`document.getElementById('step').click()`);
    await wait(`document.getElementById('status').textContent==='Paused'&&document.getElementById('epoch').textContent==='1'`);
    assert.ok(await evaluate(`document.getElementById('train').textContent.includes('Continue training')&&document.getElementById('channel-select').max==='${width - 1}'&&document.getElementById('model-detail').textContent.includes('${parameters.toLocaleString('en-US')}')`));
    assert.equal(await download(name), source, 'Learning must not put trained weights in the source export.');
    records.push({model, preset, width, parameters, sourceDownload: true, sourceUnchangedAfterTraining: true, firstEpoch: await evaluate(row)});
    console.log(`PASS ${model} ${preset}: real .py download, one browser epoch, correct width and preserved source.`);
  }
  // The fast W128 preset reaches 99% in a few dozen epochs.
  await evaluate(`document.querySelector('[data-model-tab="dense-rotary"]').click();document.querySelector('[data-preset-tab="fast"]').click()`);
  await wait(`document.getElementById('epoch').textContent==='1'&&document.getElementById('status').textContent==='Paused'`);
  await evaluate(`document.getElementById('limit').value='10000';document.getElementById('limit').dispatchEvent(new Event('change'));document.getElementById('capture').value='1';document.getElementById('capture').dispatchEvent(new Event('change'));document.getElementById('train').click()`);
  await wait(`document.getElementById('status').textContent==='Paused at 99% held-out accuracy'`, 300000);
  const paused = await evaluate<{epoch: number; accuracy: number; observations: string; workers: number}>(row);
  assert.ok(paused.accuracy >= 99 && paused.epoch > 1 && paused.epoch < 10000);
  assert.ok(await evaluate(`document.getElementById('train').textContent.includes('Continue training')&&!document.getElementById('train').disabled`));
  console.log(`PASS auto-pause: epoch ${paused.epoch}, test ${paused.accuracy}%.`);
  await writeFile('/tmp/grokking-auto-pause.png', Buffer.from((await client.call<{data: string}>('Page.captureScreenshot')).data, 'base64'));
  await evaluate(`document.getElementById('limit').value='${paused.epoch + 3}';document.getElementById('limit').dispatchEvent(new Event('change'));document.getElementById('train').click()`);
  await wait(`document.getElementById('status').textContent==='Epoch limit reached'&&document.getElementById('epoch').textContent==='${paused.epoch + 3}'`);
  const continued = await evaluate<{epoch: number; workers: number; observations: string}>(row);
  assert.equal(continued.workers, paused.workers, 'Continue must reuse the same training worker and optimizer.');
  assert.equal(parseInt(continued.observations), parseInt(paused.observations) + 3);
  await evaluate(`document.getElementById('reset').click()`);
  await wait(`document.getElementById('status').textContent==='Ready'&&document.getElementById('epoch').textContent==='0'`);
  assert.equal(await evaluate(`document.getElementById('observations').textContent`), '1 observations');
  if (client.errors.length) throw new Error(JSON.stringify(client.errors));
  await client.call('Page.removeScriptToEvaluateOnNewDocument', {identifier: injection.identifier});
  await writeFile('/tmp/grokking-export-browser.json', JSON.stringify({records, paused, continued, reset: true, errors: client.errors}, null, 2) + '\n');
  console.log('PASS: mandatory target pause, Continue preserves worker/optimizer/history, three more epochs complete without re-pausing, and Reset restores the initial state.');
} finally {client.close();}
