import assert from 'node:assert/strict';
import {CDP} from './cdp.ts';

const client = await CDP.connect(Number(process.env.GROKKING_CDP_PORT || 9224));
const origin = (process.env.GROKKING_PREVIEW_ORIGIN || 'http://127.0.0.1:8081/grokking_abmod').replace(/\/$/, '');
const evaluate = client.evaluate.bind(client), wait = client.wait.bind(client);
try {
  await client.call('Runtime.enable'); await client.call('Page.enable');
  await client.call('Emulation.setDeviceMetricsOverride', {width: 1366, height: 768, deviceScaleFactor: 1, mobile: false});
  await client.call('Page.navigate', {url: origin + '/?lang=en'});
  await wait(`document.getElementById('task-dialog')?.open&&document.getElementById('status')?.textContent==='Ready'`);
  assert.ok(await evaluate(`crossOriginIsolated&&navigator.serviceWorker.controller&&typeof SharedArrayBuffer==='function'`), 'Static host must support shared CPU workers after the bootstrap reload.');
  const assets = await evaluate<{modules: number; license: string; unsupported: number}>(`(async()=>{
    const {MODEL_IDS}=await import('./types.js');let modules=0;
    for(const model of MODEL_IDS)for(const variant of ['scalar','simd'])for(const shared of [false,true]){
      const response=await fetch('./'+model+'-'+variant+(shared?'-shared':'')+'.wasm');
      if(!response.ok||!WebAssembly.validate(await response.arrayBuffer()))throw new Error('Missing WASM asset');modules++;
    }
    return {modules,license:await(await fetch('./LICENSE.txt')).text(),unsupported:(await fetch('./dense-rotary-48-simd.wasm')).status};
  })()`);
  assert.equal(assets.modules, 20); assert.equal(assets.unsupported, 404);
  assert.ok(assets.license.startsWith('MIT License\n') && assets.license.includes('Mikhail Borisov'));
  await evaluate(`document.getElementById('task-open').click();document.getElementById('backend').value='simd-4';document.getElementById('backend').dispatchEvent(new Event('change'))`);
  await wait(`document.getElementById('status').textContent==='Ready'&&document.getElementById('device').textContent.includes('4 threads')`);
  for (const width of [16, 32, 64, 128, 256]) {
    await evaluate(`document.querySelector('[data-view-tab="settings"]').click();for(const [id,value] of [['custom-width','${width}'],['custom-lr','0.0025'],['custom-wd','0']]){const input=document.getElementById(id);input.value=value;input.dispatchEvent(new Event('input'));}document.getElementById('settings-form').requestSubmit()`);
    await wait(`document.getElementById('status').textContent==='Ready'&&document.getElementById('model-detail').textContent.startsWith('${width} channels')`);
    assert.ok(await evaluate(`document.getElementById('device').textContent.includes('4 threads')&&document.getElementById('model-optimizer').textContent==='lr=0.0025 · wd=0'`));
    await evaluate(`document.getElementById('step').click()`);
    await wait(`document.getElementById('epoch').textContent==='1'&&document.getElementById('status').textContent==='Paused'`);
    assert.ok(await evaluate(`document.getElementById('error').hidden&&!/NaN|Infinity/.test(document.getElementById('train-loss').textContent)`));
    console.log(`Static Pages: W${width}, four threads, custom AdamW, one epoch passed.`);
  }
  assert.deepEqual(client.errors, []);
  console.log('PASS: project-prefixed static URLs, isolation worker, all 20 prebuilt WASM modules, five custom widths, four-thread training and MIT license.');
} finally {client.close();}
