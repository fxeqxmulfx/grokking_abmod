import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {CDP} from './cdp.ts';

const client = await CDP.connect(Number(process.env.GROKKING_CDP_PORT || 9224));
const origin = (process.env.GROKKING_PREVIEW_ORIGIN || 'http://127.0.0.1:8080').replace(/\/$/, '');
const evaluate = client.evaluate.bind(client), wait = client.wait.bind(client);
const click = (selector: string) => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
const paint = () => evaluate('document.fonts.ready.then(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))))');
const ready = () => wait(`!document.getElementById('step').disabled&&['Ready','Paused','Epoch limit reached'].includes(document.getElementById('status').textContent)`);
const digest = () => evaluate<string>(`crypto.subtle.digest('SHA-256',window.memoryReview.weights).then(bytes=>Array.from(new Uint8Array(bytes),value=>value.toString(16).padStart(2,'0')).join(''))`);
async function workerCount() {
  const {targetInfos} = await client.call<{targetInfos: {type: string}[]}>('Target.getTargets');
  return targetInfos.filter(target => target.type === 'worker').length;
}
async function workers(expected: number) {
  const deadline = Date.now() + 5000;
  while (await workerCount() !== expected && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 50));
  assert.equal(await workerCount(), expected, 'Release all workers belonging to hidden, paused views.');
}
interface Storage {pixels: number; hiddenPixels: number; rasterBytes: number; rasterPixels: number; view: string}
async function storage() {
  await paint();
  const value = await evaluate<Storage>(`(() => {
    let pixels=0,hiddenPixels=0;
    for(const canvas of document.querySelectorAll('main canvas')){const rect=canvas.getBoundingClientRect(),size=canvas.width*canvas.height;pixels+=size;if(!rect.width||!rect.height)hiddenPixels+=size;}
    const data=document.getElementById('mesh-edges').dataset;
    return{pixels,hiddenPixels,rasterBytes:Number(data.memoryBytes||0),rasterPixels:Number(data.rasterPixels||0),view:document.body.dataset.view};
  })()`);
  assert.ok(value.hiddenPixels <= 7, 'Hidden canvases must retain only 1×1 placeholder surfaces.');
  assert.ok(value.pixels < 800000, 'A mobile view must stay within its total display-surface budget.');
  assert.ok(value.rasterBytes <= 4 * 1024 * 1024, 'Bound the mobile network accumulation buffer.');
  return value;
}
let injected: {identifier: string} | undefined;
try {
  await client.call('Page.enable'); await client.call('Runtime.enable');
  await client.call('Emulation.setDeviceMetricsOverride', {width: 412, height: 915, deviceScaleFactor: 3, mobile: true});
  await client.call('Emulation.setTouchEmulationEnabled', {enabled: true, maxTouchPoints: 5});
  await client.call('Emulation.setUserAgentOverride', {userAgent: 'Mozilla/5.0 (Linux; Android 14; SM-A235F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36'});
  await client.call('Emulation.setCPUThrottlingRate', {rate: 6});
  injected = await client.call<{identifier: string}>('Page.addScriptToEvaluateOnNewDocument', {source: `(() => {
    window.memoryReview={weights:null};const NativeWorker=Worker;
    window.Worker=class extends NativeWorker{constructor(...args){super(...args);this.addEventListener('message',({data})=>{if(data.type==='weights')window.memoryReview.weights=data.weights.slice().buffer})}};
  })()`});
  await client.call('Page.navigate', {url: origin + '/?lang=en'});
  await wait(`document.getElementById('task-dialog')?.open&&document.getElementById('status').textContent==='Ready'`);
  await click('#task-open'); await workers(2);
  // Complete exactly fifteen W128 epochs, ending in the same paused state.
  for (let epoch = 1; epoch <= 15; epoch++) {
    await click('#step'); await wait(`document.getElementById('epoch').textContent==='${epoch}'&&document.getElementById('status').textContent==='Paused'`);
  }
  const before = await digest();
  const history = await evaluate(`document.getElementById('observations').textContent`);
  const records: Storage[] = [];
  for (let i = 0; i < 12; i++) {
    await click('[data-view-tab="channels"]');
    await wait(`(() => {const d=document.getElementById('mesh-edges').dataset;return d.frame&&d.frame===d.request})()`);
    await workers(1); records.push(await storage());
    assert.equal(await evaluate(`document.getElementById('mesh-edges').dataset.connections`), '53632');
    await click('[data-view-tab="weights"]'); await workers(0);
    const weights = await storage(); records.push(weights);
    assert.equal(weights.rasterBytes, 0); assert.equal(weights.rasterPixels, 0);
    assert.ok(await evaluate(`['input-a','input-b','readout'].every(id=>{const c=document.getElementById(id),p=c.getContext('2d').getImageData(Math.floor(c.width/2),Math.floor(c.height/2),1,1).data;return p[3]===255})`), 'Returning to the atlas must paint all three heatmaps.');
    assert.equal(await evaluate(`document.getElementById('epoch').textContent`), '15');
    assert.equal(await evaluate(`document.getElementById('observations').textContent`), history);
  }
  await evaluate(`(() => {for(let i=0;i<40;i++){document.querySelector('[data-view-tab="channels"]').click();document.querySelector('[data-view-tab="weights"]').click()}})()`);
  await workers(0); records.push(await storage());
  assert.equal(await digest(), before, 'Switching views must preserve every trained weight.');
  await click('[data-view-tab="training"]'); await ready(); await workers(2);
  assert.equal(await evaluate(`document.getElementById('epoch').textContent`), '15');
  assert.equal(await evaluate(`document.getElementById('observations').textContent`), history);
  await click('#step'); await wait(`document.getElementById('epoch').textContent==='16'&&document.getElementById('status').textContent==='Paused'`);
  const resumed = await digest();
  await click('#reset'); await ready();
  for (let epoch = 1; epoch <= 16; epoch++) {
    await click('#step'); await wait(`document.getElementById('epoch').textContent==='${epoch}'&&document.getElementById('status').textContent==='Paused'`);
  }
  assert.equal(await digest(), resumed, 'Freeing the paused trainer must preserve AdamW and sampler state bit-for-bit.');
  // Switching immediately after Pause must wait for worker shutdown before
  // allocating the next view, even if the last training chunk is in flight.
  await click('#train'); await wait(`document.getElementById('status').textContent==='Training'`);
  await evaluate(`document.getElementById('train').click();document.querySelector('[data-view-tab="channels"]').click();document.querySelector('[data-view-tab="weights"]').click()`);
  await workers(0); records.push(await storage());
  assert.ok(Number(await evaluate(`document.getElementById('epoch').textContent`)) >= 16);
  assert.deepEqual(client.errors, []);
  await writeFile('/tmp/grokking-mobile-memory-verification.json', JSON.stringify({origin,epochsBeforeSwitching:15,records,bitwiseContinuation:true,errors:client.errors},null,2)+'\n');
  console.log(`PASS: W128 at epoch 15, Network → Weights, 12 complete render/release cycles, 80 rapid switches, immediate Pause → Network → Weights, no hidden pixel buffers, zero workers on Weights, bounded raster memory and bit-for-bit epoch 16 after restoring training.`);
} finally {
  if (injected) await client.call('Page.removeScriptToEvaluateOnNewDocument', {identifier: injected.identifier});
  await client.call('Emulation.setCPUThrottlingRate', {rate: 1});
  await client.call('Emulation.setUserAgentOverride', {userAgent: ''});
  await client.call('Emulation.setTouchEmulationEnabled', {enabled: false});
  client.close();
}
