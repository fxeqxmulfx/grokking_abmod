import assert from 'node:assert/strict';
import {CDP} from './cdp.ts';

const client = await CDP.connect(Number(process.env.GROKKING_CDP_PORT || 9224));
const origin = process.env.GROKKING_PREVIEW_ORIGIN || 'http://127.0.0.1:8080';
const evaluate = client.evaluate.bind(client), wait = client.wait.bind(client);
const click = (selector: string) => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
const paint = () => evaluate('document.fonts.ready.then(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))))');
const ready = () => wait(`!document.getElementById('step').disabled&&['Ready','Paused','Epoch limit reached'].includes(document.getElementById('status').textContent)`);
const digest = () => evaluate<string>(`crypto.subtle.digest('SHA-256',window.mobileReview.weights).then(bytes=>Array.from(new Uint8Array(bytes),value=>value.toString(16).padStart(2,'0')).join(''))`);
async function workers() {
  const {targetInfos} = await client.call<{targetInfos: {type: string}[]}>('Target.getTargets');
  return targetInfos.filter(target => target.type === 'worker').length;
}
async function boundedWorkers() {
  const deadline = Date.now() + 3000;
  while (await workers() > 3 && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 50));
  assert.ok(await workers() <= 3, 'Only one two-thread mobile trainer and one network renderer may remain alive.');
}
let switches = 0;
try {
  await client.call('Page.enable'); await client.call('Runtime.enable');
  await client.call('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 3, mobile: true});
  await client.call('Emulation.setTouchEmulationEnabled', {enabled: true, maxTouchPoints: 5});
  await client.call('Emulation.setUserAgentOverride', {userAgent: 'Mozilla/5.0 (Linux; Android 14; SM-A235F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36', platform: 'Linux armv8l'});
  await client.call('Emulation.setCPUThrottlingRate', {rate: 6});
  const injected = await client.call<{identifier: string}>('Page.addScriptToEvaluateOnNewDocument', {source: `(() => {
    window.mobileReview={weights:null,labels:[]};
    const NativeWorker=Worker;
    window.Worker=class extends NativeWorker {constructor(...args){super(...args);this.addEventListener('message',({data})=>{if(data.type==='weights')window.mobileReview.weights=data.weights.slice().buffer})}};
    const fill=CanvasRenderingContext2D.prototype.fillText,rect=CanvasRenderingContext2D.prototype.fillRect,clear=CanvasRenderingContext2D.prototype.clearRect;
    CanvasRenderingContext2D.prototype.clearRect=function(...args){if(this.canvas.id==='mesh')window.mobileReview.labels=[];return clear.apply(this,args)};
    CanvasRenderingContext2D.prototype.fillRect=function(x,y,w,h){if(this.canvas.id==='mesh')window.mobileReview.background={left:x,top:y,right:x+w,bottom:y+h};return rect.call(this,x,y,w,h)};
    CanvasRenderingContext2D.prototype.fillText=function(text,x,y,...args){if(this.canvas.id==='mesh'&&String(text).includes('ReLU')&&String(text).includes('RMSNorm')){const m=this.measureText(text);window.mobileReview.labels.push({text,background:window.mobileReview.background,bounds:{left:x-m.actualBoundingBoxLeft,right:x+m.actualBoundingBoxRight,top:y-m.actualBoundingBoxAscent,bottom:y+m.actualBoundingBoxDescent}})}return fill.call(this,text,x,y,...args)};
  })()`});
  await client.call('Page.navigate', {url: origin + '/?lang=en'});
  await wait(`document.getElementById('task-dialog')?.open&&document.getElementById('status').textContent==='Ready'`);
  await click('#task-open');
  assert.match(await evaluate<string>(`document.getElementById('device').textContent`), /2 threads/, 'Automatic mobile training must reserve CPU capacity for the UI.');
  for (const model of ['dense-rotary-16', 'dense-rotary-32', 'dense-rotary-64', 'dense-rotary', 'dense-rotary-256']) {
    await click(`[data-model-tab="${model}"]`); await ready();
    const presets = await evaluate<string[]>(`Array.from(document.querySelectorAll('[data-preset-tab]')).filter(button=>!button.hidden).map(button=>button.dataset.presetTab)`);
    for (const preset of presets) {
      await click(`[data-preset-tab="${preset}"]`); await ready();
      for (const view of ['channels', 'weights', 'settings', 'training']) {await click(`[data-view-tab="${view}"]`); await paint(); switches++;}
      await boundedWorkers();
    }
  }
  // Switching away during startup must coalesce requests into the last model.
  await evaluate(`(() => {for(let i=0;i<30;i++){document.querySelector('[data-model-tab="dense-rotary-256"]').click();document.querySelector('[data-model-tab="dense-rotary-16"]').click()}document.querySelector('[data-model-tab="dense-rotary-32"]').click();document.querySelector('[data-preset-tab="fast"]').click()})()`);
  await ready(); await boundedWorkers();
  await evaluate(`document.getElementById('backend').value='simd';document.getElementById('backend').dispatchEvent(new Event('change'))`);
  await ready();
  await click('#step'); await wait(`document.getElementById('epoch').textContent==='1'&&!document.getElementById('step').disabled`);
  const loss = await evaluate<string>(`document.getElementById('train-loss').textContent`);
  await click('[data-model-tab="dense-rotary-64"]'); await ready();
  await click('[data-model-tab="dense-rotary-32"]'); await ready();
  assert.equal(await evaluate(`document.getElementById('epoch').textContent`), '1');
  assert.equal(await evaluate(`document.getElementById('train-loss').textContent`), loss);
  assert.equal(await evaluate(`document.getElementById('observations').textContent`), '2 observations', 'Resuming must not duplicate an evaluation.');
  await click('#step'); await wait(`document.getElementById('epoch').textContent==='2'&&!document.getElementById('step').disabled`);
  const resumed = await digest();
  await click('#reset'); await ready();
  for (const epoch of [1, 2]) {await click('#step'); await wait(`document.getElementById('epoch').textContent==='${epoch}'&&!document.getElementById('step').disabled`);}
  assert.equal(await digest(), resumed, 'The optimizer and shuffled sampler must resume bit-for-bit after switching models.');
  // Suspend while training is in progress, then continue the same experiment.
  await click('#train'); await wait(`document.getElementById('status').textContent==='Training'`);
  await click('[data-model-tab="dense-rotary-16"]'); await ready(); await boundedWorkers();
  await click('[data-model-tab="dense-rotary-32"]'); await ready();
  assert.ok(Number(await evaluate(`document.getElementById('epoch').textContent`)) >= 2);
  for (const [width, height] of [[320, 568], [390, 844], [412, 915], [844, 390]]) {
    await client.call('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: 3, mobile: true});
    for (let i = 0; i < 12; i++) for (const view of ['channels', 'weights', 'settings', 'training']) {
      await click(`[data-view-tab="${view}"]`); await paint(); switches++;
    }
  }
  await client.call('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 3, mobile: true});
  await click('[data-view-tab="channels"]');
  for (const locale of ['en', 'es', 'de', 'ja', 'fr', 'ru']) {
    await evaluate(`document.getElementById('language').value='${locale}';document.getElementById('language').dispatchEvent(new Event('change'))`);
    await wait(`document.documentElement.lang==='${locale}'`); await paint();
    assert.ok(await evaluate(`window.mobileReview.labels.length>0&&window.mobileReview.labels.every(({background:b,bounds:r})=>b&&r.left>=b.left+1&&r.right<=b.right-1&&r.top>=b.top&&r.bottom<=b.bottom)`), `The complete ${locale} Dense/ReLU/skip/RMSNorm label must fit on its background.`);
  }
  await boundedWorkers();
  assert.deepEqual(client.errors, []);
  await client.call('Page.removeScriptToEvaluateOnNewDocument', {identifier: injected.identifier});
  console.log(`PASS: Android Chrome, 6× CPU slowdown, DPR 3, ${switches} view switches, all widths/presets, rapid startup switches, bounded workers, suspend during training, bit-for-bit continuation and six complete architecture captions.`);
} finally {
  await client.call('Emulation.setCPUThrottlingRate', {rate: 1});
  await client.call('Emulation.setUserAgentOverride', {userAgent: ''});
  await client.call('Emulation.setTouchEmulationEnabled', {enabled: false});
  client.close();
}
