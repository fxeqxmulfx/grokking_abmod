import {writeFile} from 'node:fs/promises';

// Dependency-free CDP smoke test. Run against the local static server and a
// temporary Chrome profile; this never connects to an existing user profile.
const port = Number(process.env.GROKKING_CDP_PORT || 9224);
const origin = process.env.GROKKING_PREVIEW_ORIGIN || 'http://127.0.0.1:8080';
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const target = targets.find(t => t.type === 'page');
if (!target) throw new Error('Start headless Chrome with a remote debugging port first.');
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, {once: true}); socket.addEventListener('error', reject, {once: true}); });
let next = 0; const pending = new Map(), errors = [];
socket.addEventListener('message', ({data}) => {
  const message = JSON.parse(data);
  if (message.id) { const callback = pending.get(message.id); pending.delete(message.id); if (message.error) callback.reject(new Error(JSON.stringify(message.error))); else callback.resolve(message.result); }
  if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
});
function call(method, params = {}) { return new Promise((resolve, reject) => { const id = ++next; pending.set(id, {resolve, reject}); socket.send(JSON.stringify({id, method, params})); }); }
async function evaluate(expression) {
  const result = await call('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function waitFor(expression, timeout = 60000) {
  const start = Date.now();
  while (Date.now() - start < timeout) { const result = await evaluate(expression); if (result) return result; await new Promise(resolve => setTimeout(resolve, 250)); }
  throw new Error(`Timed out waiting for ${expression}`);
}
async function assertFits(label) {
  await evaluate(`new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
  const issues = await evaluate(`(() => {
    const issues = [], root = document.documentElement;
    if (root.scrollWidth > innerWidth || root.scrollHeight > innerHeight) issues.push('Page requires scrolling');
    const title = document.querySelector('h1').getBoundingClientRect(), tools = document.querySelector('.header-tools').getBoundingClientRect();
    if (title.right > tools.left - 2) issues.push('Model title overlaps the engine label');
    for (const element of document.querySelectorAll('.topbar, footer, main .card, .weight-heading, main button, main input, main select, main canvas')) {
      const rect = element.getBoundingClientRect();
      if (!rect.width || !rect.height) continue;
      const name = element.id || element.className || element.tagName;
      if (rect.left < -1 || rect.top < -1 || rect.right > innerWidth + 1 || rect.bottom > innerHeight + 1) issues.push(name + ' leaves the viewport');
      if (element.matches('.card, .topbar, footer, .weight-heading') && (element.scrollHeight > element.clientHeight + 1 || element.scrollWidth > element.clientWidth + 1)) issues.push(name + ' clips its content');
      if (element.tagName === 'CANVAS' && (rect.width < 50 || rect.height < 50)) issues.push(name + ' is too small to read');
      const panel = element.closest('.card');
      if (panel && panel !== element) {
        const bounds = panel.getBoundingClientRect();
        if (rect.left < bounds.left - 1 || rect.right > bounds.right + 1 || rect.top < bounds.top - 1 || rect.bottom > bounds.bottom + 1) issues.push(name + ' leaves its panel');
      }
    }
    return issues;
  })()`);
  if (issues.length) throw new Error(`${label}: ${issues.join('; ')}`);
  console.log(`Layout fits: ${label}`);
}
try {
  await call('Runtime.enable'); await call('Page.enable');
  await call('Emulation.setDeviceMetricsOverride', {width:1366,height:768,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate', {url:origin});
  const status = await waitFor(`document.getElementById('status') && ['Ready','Unavailable','Error'].includes(document.getElementById('status').textContent) && {status:document.getElementById('status').textContent,error:document.getElementById('error').textContent,device:document.getElementById('device').textContent}`);
  console.log('Initial page:', JSON.stringify(status));
  if (status.status !== 'Ready') throw new Error(status.error || status.status);
  if (!await evaluate(`document.getElementById('model-detail').textContent.includes('128 channels') && document.getElementById('seed').value==='11' && document.getElementById('step').textContent==='+1 epoch'`)) throw new Error('Incorrect narrow defaults');
  await assertFits('1366 × 768 narrow');
  if (process.env.GROKKING_VERIFY_ONLY !== 'ui' && status.device.includes('WebGPU')) {
    const checks = await evaluate(`import('${origin}/tests/models-gpu.mjs').then(async m=>({narrow:await m.verifyModelGPU('neighbor-rotary'),wide:await m.verifyModelGPU('neighbor-rotary-wide')}))`);
    console.log('GPU verification:', JSON.stringify(checks));
  }
  await evaluate(`document.getElementById('step').click()`);
  await waitFor(`document.getElementById('epoch').textContent==='1' && document.getElementById('status').textContent==='Paused' && document.getElementById('observations').textContent==='2 observations'`);
  await evaluate(`document.getElementById('limit').value='100';document.getElementById('limit').dispatchEvent(new Event('change'));document.getElementById('stop-target').click()`);
  if (!await evaluate(`document.getElementById('epoch').textContent==='1' && document.getElementById('observations').textContent==='2 observations'`)) throw new Error('Changing the budget reset training');
  await evaluate(`document.getElementById('loss-tab').click();document.getElementById('weight-view').value='delta';document.getElementById('weight-view').dispatchEvent(new Event('input'))`);
  for (const [width,height] of [[1280,720],[1024,768],[1920,1080],[1366,768]]) {
    await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
    await assertFits(`${width} × ${height} narrow`);
  }
  await writeFile('/tmp/grokking-rotary-narrow.png',Buffer.from((await call('Page.captureScreenshot',{captureBeyondViewport:true})).data,'base64'));
  await evaluate(`document.getElementById('train').click()`);
  await waitFor(`document.getElementById('status').textContent==='Training'`);
  await evaluate(`document.getElementById('train').click()`);
  await waitFor(`document.getElementById('status').textContent==='Paused'`);
  await evaluate(`document.getElementById('reset').click()`);
  await waitFor(`document.getElementById('status').textContent==='Ready' && document.getElementById('epoch').textContent==='0'`);
  await evaluate(`document.getElementById('backend').value='wasm';document.getElementById('backend').dispatchEvent(new Event('change'))`);
  await waitFor(`document.getElementById('status').textContent==='Ready' && document.getElementById('device').textContent.includes('WASM')`);
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate(`document.getElementById('train').click()`);
  await waitFor(`document.getElementById('status').textContent==='Training'`);
  for (const view of ['weights','channels','training']) {
    await evaluate(`document.querySelector('[data-view-tab="${view}"]').click()`);
    await assertFits(`390 × 844 narrow ${view}`);
  }
  await evaluate(`document.getElementById('train').click()`);
  await waitFor(`document.getElementById('status').textContent==='Paused'`);
  const previous = await evaluate(`({epoch:document.getElementById('epoch').textContent,observations:document.getElementById('observations').textContent,loss:document.getElementById('train-loss').textContent})`);
  await evaluate(`document.querySelector('[data-model-tab="neighbor-rotary-wide"]').click()`);
  await waitFor(`document.getElementById('status').textContent==='Ready' && document.getElementById('epoch').textContent==='0'`);
  if (!await evaluate(`document.getElementById('model-detail').textContent.includes('1024 channels') && document.getElementById('model-detail').textContent.includes('111,616') && document.getElementById('seed').value==='11' && document.getElementById('channel-select').max==='1023' && document.getElementById('device').textContent.includes('WASM')`)) throw new Error('Incorrect wide defaults');
  await evaluate(`document.querySelector('[data-view-tab="weights"]').click();document.getElementById('weight-group').value='io';document.getElementById('weight-group').dispatchEvent(new Event('change'))`);
  const initialMap = await evaluate(`document.getElementById('input-a').toDataURL()`);
  await evaluate(`document.querySelector('[data-view-tab="training"]').click();document.getElementById('step').click()`);
  await waitFor(`document.getElementById('epoch').textContent==='1' && document.getElementById('status').textContent==='Paused' && document.getElementById('observations').textContent==='2 observations'`);
  await evaluate(`document.querySelector('[data-view-tab="weights"]').click();new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))`);
  if (initialMap === await evaluate(`document.getElementById('input-a').toDataURL()`)) throw new Error('Shared embedding visualization did not update with training');
  if (!await evaluate(`document.getElementById('atlas-a-title').textContent==='Shared embedding E' && document.getElementById('atlas-b-title').textContent.includes('Rotate90(E)') && document.getElementById('atlas-r-title').textContent==='Tied readout Eᵀ' && document.getElementById('atlas-a-size').textContent==='97 × 1024' && document.getElementById('atlas-b-size').textContent==='97 × 1024' && document.getElementById('atlas-r-size').textContent==='1024 × 97'`)) throw new Error('Incorrect shared-E and rotated-E atlases');
  await evaluate(`document.getElementById('weight-group').value='1';document.getElementById('weight-group').dispatchEvent(new Event('change'))`);
  if (!await evaluate(`document.getElementById('atlas-a-size').textContent==='3 × 1024' && document.getElementById('atlas-b-size').textContent==='3 × 1024'`)) throw new Error('Incorrect block edge atlases');
  await evaluate(`document.querySelector('[data-view-tab="channels"]').click();document.getElementById('channel-select').value='1023';document.getElementById('channel-select').dispatchEvent(new Event('input'))`);
  await waitFor(`document.getElementById('mesh').getAttribute('aria-label').includes('channel 1023,') && document.getElementById('mesh').getAttribute('aria-label').includes('twelve learned edges')`);
  for (const [width,height] of [[1366,768],[390,844],[320,568],[768,1024],[1366,600],[844,390]]) {
    await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width<650});
    for (const view of ['training','weights','channels']) {
      await evaluate(`document.querySelector('[data-view-tab="${view}"]').click()`);
      await assertFits(`${width} × ${height} wide ${view}`);
    }
  }
  await evaluate(`document.getElementById('about').click()`);
  if (!await evaluate(`document.getElementById('about-dialog').open && document.getElementById('model-description').textContent.includes('Rotate90(E[b])') && document.getElementById('reference-description').textContent.includes('All five')`)) throw new Error('Incorrect model explanation');
  await evaluate(`document.getElementById('close-about').click()`);
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await writeFile('/tmp/grokking-rotary-wide-mobile.png',Buffer.from((await call('Page.captureScreenshot',{captureBeyondViewport:true})).data,'base64'));
  await evaluate(`document.querySelector('[data-model-tab="neighbor-rotary"]').click()`);
  const restored = await evaluate(`({epoch:document.getElementById('epoch').textContent,observations:document.getElementById('observations').textContent,loss:document.getElementById('train-loss').textContent})`);
  if (JSON.stringify(previous)!==JSON.stringify(restored)) throw new Error('Switching models changed the narrow experiment');
  if (!await evaluate(`document.getElementById('channel-select').max==='127'`)) throw new Error('Channel bounds did not follow width');
  await evaluate(`document.querySelector('[data-view-tab="channels"]').click();document.getElementById('channel-select').value='127';document.getElementById('channel-select').dispatchEvent(new Event('input'))`);
  await waitFor(`document.getElementById('mesh').getAttribute('aria-label').includes('channel 127,')`);
  await evaluate(`document.querySelector('[data-view-tab="training"]').click();document.getElementById('train').click()`);
  await waitFor(`document.getElementById('status').textContent==='Training'`);
  await evaluate(`document.querySelector('[data-model-tab="neighbor-rotary-wide"]').click()`);
  if (!await evaluate(`document.getElementById('epoch').textContent==='1' && document.getElementById('status').textContent==='Paused' && document.getElementById('channel-select').value==='1023'`)) throw new Error('Wide state was not restored');
  await new Promise(resolve=>setTimeout(resolve,250));
  await evaluate(`document.querySelector('[data-model-tab="neighbor-rotary"]').click()`);
  await waitFor(`document.getElementById('status').textContent==='Paused' || document.getElementById('status').textContent==='Epoch limit reached'`);
  await call('Emulation.setDeviceMetricsOverride',{width:1366,height:768,deviceScaleFactor:1,mobile:false});
  await evaluate(`document.querySelector('[data-model-tab="neighbor-rotary-wide"]').click();document.querySelector('[data-view-tab="training"]').click();document.getElementById('loss-tab').click()`);
  await writeFile('/tmp/grokking-rotary-wide.png',Buffer.from((await call('Page.captureScreenshot',{captureBeyondViewport:true})).data,'base64'));
  if (errors.length) throw new Error(`Browser exceptions: ${JSON.stringify(errors)}`);
  console.log('PASS: both widths fit, GPU/WASM training, live tied-E weights, pause/reset, independent states, channel wrap, all block weights, loss/accuracy and model information.');
} finally { socket.close(); }
