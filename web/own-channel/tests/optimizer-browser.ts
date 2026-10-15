import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {CDP} from './cdp.ts';
import {modelInfo, modelPresets, optimizerProfile} from '../dist/models.js';
import {MODEL_IDS} from '../dist/types.js';

// The W128 fast preset is covered by export-browser; verify every other preset here.
const presets = MODEL_IDS.flatMap(model => modelPresets(model).map(preset => ({model, preset}))).filter(({model, preset}) => model !== 'dense-rotary' || preset !== 'fast');
const client = await CDP.connect(Number(process.env.GROKKING_CDP_PORT ?? 9224));
const origin = process.env.GROKKING_PREVIEW_ORIGIN ?? 'http://127.0.0.1:8080';
const evaluate = client.evaluate.bind(client), wait = client.wait.bind(client);
const snapshot = `({epoch:Number(document.getElementById('epoch').textContent.replaceAll(',','')),accuracy:parseFloat(document.getElementById('heldout-accuracy').textContent),observations:parseInt(document.getElementById('observations').textContent),workers:window.__optimizerWorkers.length})`;
const records: unknown[] = [];
try {
  await client.call('Runtime.enable'); await client.call('Page.enable');
  const injection = await client.call<{identifier: string}>('Page.addScriptToEvaluateOnNewDocument', {source: `localStorage.setItem('grokking.locale','en');window.__optimizerWorkers=[];const NativeWorker=Worker;window.Worker=class extends NativeWorker {constructor(...args){super(...args);window.__optimizerWorkers.push(this)}};`});
  await client.call('Emulation.setDeviceMetricsOverride', {width: 1366, height: 768, deviceScaleFactor: 1, mobile: false});
  await client.call('Page.navigate', {url: origin});
  await wait(`document.getElementById('task-dialog')?.open`);
  await evaluate(`document.getElementById('task-open').click();document.getElementById('capture').value='1';document.getElementById('capture').dispatchEvent(new Event('change'))`);
  for (const {model, preset} of presets) {
    const info = modelInfo(model), profile = optimizerProfile(model, preset);
    await evaluate(`document.querySelector('[data-model-tab="${model}"]').click();document.querySelector('[data-preset-tab="${preset}"]').click()`);
    await wait(`document.getElementById('status').textContent==='Ready'&&document.getElementById('epoch').textContent==='0'`);
    assert.ok(await evaluate(`document.getElementById('reference-description').dataset.learningRate==='${profile.learningRate}'&&document.getElementById('reference-description').dataset.weightDecay==='${profile.weightDecay}'`));
    await evaluate(`document.getElementById('limit').value='10000';document.getElementById('limit').dispatchEvent(new Event('change'));document.getElementById('capture').value='1';document.getElementById('capture').dispatchEvent(new Event('change'));document.getElementById('train').click()`);
    const start = Date.now();
    // Slow presets train for many minutes, so only an epoch counter stalled for ten minutes times out.
    const done = `['Paused at 99% held-out accuracy','Epoch limit reached','Unavailable'].includes(document.getElementById('status').textContent)`;
    const epoch = `Number(document.getElementById('epoch').textContent.replaceAll(',',''))`;
    while (!await evaluate<boolean>(done)) await wait(`${done}||${epoch}>${await evaluate<number>(epoch)}`, 600000);
    const paused = await evaluate<{epoch: number; accuracy: number; observations: number; workers: number}>(snapshot);
    assert.equal(await evaluate(`document.getElementById('status').textContent`), 'Paused at 99% held-out accuracy', `${model} ${preset}: the preset did not reach 99% within 10,000 epochs.`);
    assert.ok(paused.accuracy >= 99);
    const wallSeconds = (Date.now() - start) / 1000;
    console.log(`PASS ${model} ${preset}: auto-pause at epoch ${paused.epoch}, test ${paused.accuracy}%.`);
    await evaluate(`document.getElementById('limit').value='${paused.epoch + 3}';document.getElementById('limit').dispatchEvent(new Event('change'));document.getElementById('train').click()`);
    await wait(`document.getElementById('status').textContent==='Epoch limit reached'&&document.getElementById('epoch').textContent.replaceAll(',','')==='${paused.epoch + 3}'`);
    const continued = await evaluate<{epoch: number; observations: number; workers: number}>(snapshot);
    assert.equal(continued.workers, paused.workers); assert.equal(continued.observations, paused.observations + 3);
    records.push({model, preset, width: info.width, learningRate: profile.learningRate, weightDecay: profile.weightDecay, nativeSeed11Epoch: profile.nativeTest99Epochs[0], paused, continued, wallSeconds});
    console.log(`PASS ${model} ${preset}: test ${paused.accuracy}% at epoch ${paused.epoch}; Continue keeps optimizer and completes three more epochs.`);
    await writeFile('/tmp/grokking-optimizer-browser.json', JSON.stringify({records, errors: client.errors}, null, 2) + '\n');
  }
  await client.call('Page.removeScriptToEvaluateOnNewDocument', {identifier: injection.identifier});
  assert.deepEqual(client.errors, []);
} finally {client.close();}
