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
interface AtlasPixels {id: string; width: number; height: number; scale: number; opaque: number; total: number; digest: string}
const atlasRead = `['input-a','input-b','readout'].map(id=>{
  const canvas=document.getElementById(id),rect=canvas.getBoundingClientRect(),pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
  let opaque=0,hash=2166136261;for(let i=0;i<pixels.length;i++){hash=Math.imul(hash^pixels[i],16777619);if(i%4===3&&pixels[i]===255)opaque++;}
  return{id,width:canvas.width,height:canvas.height,scale:canvas.width/rect.width,opaque,total:pixels.length/4,digest:(hash>>>0).toString(16)};
})`;
async function atlasPixels() {
  const maps = await evaluate<AtlasPixels[]>(atlasRead);
  for (const map of maps) {
    assert.ok(map.scale >= 1.95, 'Phone heatmaps must render at twice their CSS resolution.');
    assert.equal(map.opaque, map.total, 'Paint every pixel of all three weight maps.');
  }
  return maps;
}
async function storage() {
  await paint();
  const value = await evaluate<Storage>(`(() => {
    let pixels=0,hiddenPixels=0;
    for(const canvas of document.querySelectorAll('main canvas')){const rect=canvas.getBoundingClientRect(),size=canvas.width*canvas.height;pixels+=size;if(!rect.width||!rect.height)hiddenPixels+=size;}
    const data=document.getElementById('mesh-edges').dataset;
    return{pixels,hiddenPixels,rasterBytes:Number(data.memoryBytes||0),rasterPixels:Number(data.rasterPixels||0),view:document.body.dataset.view};
  })()`);
  assert.ok(value.hiddenPixels <= 7, 'Hidden canvases must retain only 1×1 placeholder surfaces.');
  assert.ok(value.pixels < 1200000, 'A sharp mobile view must stay within its total display-surface budget.');
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
    window.memoryReview={weights:null,holdSuspensions:false,pending:[]};const NativeWorker=Worker;
    window.Worker=class extends NativeWorker{
      constructor(...args){super(...args);this.addEventListener('message',({data})=>{if(data.type==='weights')window.memoryReview.weights=data.weights.slice().buffer})}
      postMessage(message,...args){
        if(message.type==='suspend'&&window.memoryReview.holdSuspensions)window.memoryReview.pending.push(()=>super.postMessage(message,...args));
        else super.postMessage(message,...args);
      }
    };
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
  // Keep shutdown pending until explicitly released. The graph must wait for
  // its memory, while all weight maps must already paint from retained weights.
  await evaluate('window.memoryReview.holdSuspensions=true');
  await click('[data-view-tab="channels"]'); await paint(); await workers(2);
  assert.equal(await evaluate(`document.getElementById('mesh-edges').dataset.thread`), 'inactive');
  await click('[data-view-tab="weights"]'); await paint();
  const expectedMaps = await atlasPixels();
  assert.equal(await evaluate(`getComputedStyle(document.getElementById('input-a')).imageRendering`), 'pixelated');
  assert.equal(await evaluate('window.memoryReview.pending.length'), 1);
  await evaluate('window.memoryReview.holdSuspensions=false;for(const resume of window.memoryReview.pending.splice(0))resume()');
  await workers(0);
  for (let i = 0; i < 12; i++) {
    await click('[data-view-tab="channels"]');
    await wait(`(() => {const d=document.getElementById('mesh-edges').dataset;return d.frame&&d.frame===d.request})()`);
    await workers(1); const network = await storage(); records.push(network);
    assert.ok(network.rasterPixels > 262144, 'The mobile network must retain a sharp raster rather than the old low-resolution image.');
    assert.equal(await evaluate(`document.getElementById('mesh-edges').dataset.connections`), '53632');
    await click('[data-view-tab="weights"]'); await workers(0);
    const weights = await storage(); records.push(weights);
    assert.equal(weights.rasterBytes, 0); assert.equal(weights.rasterPixels, 0);
    assert.deepEqual(await atlasPixels(), expectedMaps, 'Returning to Weights must reproduce every heatmap pixel.');
    assert.equal(await evaluate(`document.getElementById('epoch').textContent`), '15');
    assert.equal(await evaluate(`document.getElementById('observations').textContent`), history);
  }
  const rapidMaps = await evaluate<AtlasPixels[][]>(`(async () => {
    const records=[];
    for(let i=0;i<120;i++){
      const view=['channels','weights','training','weights'][i%4];document.querySelector('[data-view-tab="'+view+'"]').click();
      await new Promise(resolve=>setTimeout(resolve,[0,16,32,70,180,40,350][i%7]));
      if(view==='weights'){
        await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
        records.push(${atlasRead});
      }
    }
    return records;
  })()`);
  for (const maps of rapidMaps) assert.deepEqual(maps, expectedMaps, 'Interleaving view switches with rendering and trainer startup must never leave a blank atlas.');
  await workers(0); records.push(await storage());
  // A restored 2D context has an empty backing store, even if its data keys
  // still match. Simulate that browser lifecycle event without changing weights.
  await evaluate(`for(const id of ['input-a','input-b','readout']){const c=document.getElementById(id);c.getContext('2d').reset();c.dispatchEvent(new Event('contextrestored'));}`);
  await paint(); assert.deepEqual(await atlasPixels(), expectedMaps, 'Context restoration must invalidate the display cache.');
  const screenshot = await client.call<{data: string}>('Page.captureScreenshot', {format: 'png'});
  await writeFile('/tmp/grokking-mobile-weights.png', Buffer.from(screenshot.data, 'base64'));
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
  await writeFile('/tmp/grokking-mobile-memory-verification.json', JSON.stringify({origin,epochsBeforeSwitching:15,records,maps:expectedMaps,rapidPaints:rapidMaps.length,bitwiseContinuation:true,errors:client.errors},null,2)+'\n');
  console.log(`PASS: W128 at epoch 15, sharp heatmaps before trainer shutdown, 12 complete render/release cycles, 120 interleaved view switches with exact pixel comparisons, context restoration, immediate Pause → Network → Weights, no hidden pixel buffers, zero workers on Weights, bounded raster memory and bit-for-bit epoch 16 after restoring training.`);
} finally {
  if (injected) await client.call('Page.removeScriptToEvaluateOnNewDocument', {identifier: injected.identifier});
  await client.call('Emulation.setCPUThrottlingRate', {rate: 1});
  await client.call('Emulation.setUserAgentOverride', {userAgent: ''});
  await client.call('Emulation.setTouchEmulationEnabled', {enabled: false});
  client.close();
}
