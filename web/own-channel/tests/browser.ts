import {CDP} from './cdp.ts';
import {root} from '../project.ts';
import {writeFile} from 'node:fs/promises';
import {optimizerProfile} from '../dist/models.js';
import {MODEL_IDS} from '../dist/types.js';

// Dependency-free CDP smoke test. Run against the local static server and a
// temporary Chrome profile; this never connects to an existing user profile.
const port = Number(process.env.GROKKING_CDP_PORT || 9224);
const origin = process.env.GROKKING_PREVIEW_ORIGIN || 'http://127.0.0.1:8080';
const client = await CDP.connect(port);
const call = client.call.bind(client), evaluate = client.evaluate.bind(client), errors = client.errors;
const waitFor = client.wait.bind(client), wait = waitFor;
let layouts = 0;
async function networkPaint() {
  await evaluate(`new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))`);
  await waitFor(`(() => {const d=document.getElementById('mesh-edges').dataset;return d.frame&&d.frame===d.request})()`);
  await evaluate(`new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))`);
}
const pixelReader = `const image=new Image();image.src=canvas.toDataURL();await image.decode();const copy=document.createElement('canvas');copy.width=canvas.width;copy.height=canvas.height;const ctx=copy.getContext('2d',{willReadFrequently:true});ctx.drawImage(image,0,0);const pixels=ctx.getImageData(0,0,copy.width,copy.height).data;`;
async function connectionBands(channels: number) {
  await networkPaint();
  return evaluate(`(async () => {const {topologyLayout}=await import('${origin}/topology.js'),canvas=document.getElementById('mesh-edges'),rect=canvas.getBoundingClientRect(),rows=topologyLayout(rect.width,rect.height,${channels}).rows;${pixelReader}return [0,1,2].every(band=>{const y=Math.floor((rows[band]+rows[band+1])/2*canvas.height/rect.height);let visible=0;for(let x=0;x<canvas.width;x++)if(pixels[(y*canvas.width+x)*4+3]>0)visible++;return visible>canvas.width/4})})()`);
}
async function assertFits(label: string) {
  await evaluate(`new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
  const issues = await evaluate<string[]>(`(() => {
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
  await call('Page.addScriptToEvaluateOnNewDocument', {source: `Object.defineProperty(navigator,'gpu',{get(){throw new Error('GPU access is forbidden in the CPU-only app')}});const getContext=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(type,...args){if(/webgl|webgpu/.test(type))throw new Error('GPU canvas is forbidden');return getContext.call(this,type,...args)};`});
  await call('Emulation.setDeviceMetricsOverride', {width:1366,height:768,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate', {url:origin + '?lang=en'});
  const status = await waitFor<{status: string; error: string; device: string}>(`document.getElementById('status') && ['Ready','Unavailable','Error'].includes(document.getElementById('status').textContent) && {status:document.getElementById('status').textContent,error:document.getElementById('error').textContent,device:document.getElementById('device').textContent}`);
  console.log('Initial page:', JSON.stringify(status));
  if (status.status !== 'Ready') throw new Error(status.error || status.status);
  await waitFor(`document.getElementById('task-dialog').open && [...document.querySelectorAll('[data-task-diagram]')].every(image=>image.complete&&image.naturalWidth>0)`);
  await evaluate(`document.getElementById('task-open').click()`);
  if (!await evaluate(`document.getElementById('model-detail').textContent.includes('128 channels') && document.getElementById('model-detail').textContent.includes('28,800') && document.getElementById('model-name').textContent.includes('Dense + ReLU') && document.getElementById('seed').value==='11' && document.getElementById('step').textContent==='+1 epoch'`)) throw new Error('Incorrect narrow defaults');
  if (!await evaluate(`document.body.dataset.view==='training' && !document.getElementById('dynamics') && !document.getElementById('weight-group') && document.querySelectorAll('[data-model-tab]').length===${MODEL_IDS.length} && document.querySelectorAll('[data-view-tab]').length===4`)) throw new Error('Incorrect model tabs or obsolete workspace variants remain.');
  await waitFor(`document.getElementById('mesh').getAttribute('aria-label').includes('53632 connections')`);
  await networkPaint();
  const renderer = await evaluate<Record<string, string>>(`({...document.getElementById('mesh-edges').dataset})`);
  if (renderer.renderer !== 'CPU · Canvas 2D' || renderer.thread !== 'worker' || renderer.parameters !== '28800' || renderer.connections !== '53632') throw new Error(`Incorrect full-model drawing: ${JSON.stringify(renderer)}`);
  if (renderer.highlight !== 'false' || renderer.mode !== 'weight' || !await evaluate(`!document.getElementById('network-highlight').checked`)) throw new Error('Complete-network overview should have no channel emphasis');
  if (!await connectionBands(128)) throw new Error('Embedding, dense or readout connections are missing from the CPU drawing');
  const initialNetwork = await evaluate(`document.getElementById('mesh-edges').toDataURL()`);
  const initialScale = renderer.embeddingScale;
  await evaluate(`document.getElementById('network-view').value='delta';document.getElementById('network-view').dispatchEvent(new Event('change'))`);
  await waitFor(`document.getElementById('mesh-edges').dataset.mode==='delta'`);
  await networkPaint();
  if (!await evaluate(`(async () => {const canvas=document.getElementById('mesh-edges');${pixelReader}return !pixels.some((v,i)=>i%4===3&&v>0)})()`)) throw new Error('Change view should be zero before training');
  await evaluate(`document.getElementById('network-view').value='weight';document.getElementById('network-view').dispatchEvent(new Event('change'))`);
  await waitFor(`document.getElementById('mesh-edges').dataset.mode==='weight'`);
  await assertFits('1366 × 768 narrow');
  await evaluate(`document.getElementById('step').click()`);
  await waitFor(`document.getElementById('epoch').textContent==='1' && document.getElementById('status').textContent==='Paused' && document.getElementById('observations').textContent==='2 observations'`);
  await networkPaint();
  if (initialNetwork === await evaluate(`document.getElementById('mesh-edges').toDataURL()`)) throw new Error('Full-network edges did not update with training');
  if (initialScale !== await evaluate(`document.getElementById('mesh-edges').dataset.embeddingScale`)) throw new Error('Learning changed the overview color scale');
  const networkUploads = await evaluate(`document.getElementById('mesh-edges').dataset.uploads`);
  await evaluate(`document.getElementById('network-view').value='delta';document.getElementById('network-view').dispatchEvent(new Event('change'))`);
  await waitFor(`document.getElementById('mesh-edges').dataset.mode==='delta'`);
  if (!await connectionBands(128)) throw new Error('Weak embedding, dense or readout changes were lost after actual learning');
  await evaluate(`document.getElementById('network-highlight').click();new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))`);
  if (!await evaluate(`document.getElementById('mesh-edges').dataset.highlight==='true'&&document.getElementById('mesh-edges').dataset.connections==='53632'`)) throw new Error('Highlight hid connections');
  await evaluate(`document.getElementById('network-highlight').click();document.getElementById('network-view').value='weight';document.getElementById('network-view').dispatchEvent(new Event('change'));new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))`);
  if (networkUploads !== await evaluate(`document.getElementById('mesh-edges').dataset.uploads`)) throw new Error('Mode/highlight changes uploaded weights again');
  await evaluate(`document.getElementById('zoom-in').click()`);
  await waitFor(`document.getElementById('gain-info').textContent.includes('2.0×')`);
  await networkPaint();
  if (networkUploads !== await evaluate(`document.getElementById('mesh-edges').dataset.uploads`)) throw new Error('Zoom uploaded unchanged weights again');
  await evaluate(`document.getElementById('zoom-reset').click()`);
  await waitFor(`document.getElementById('gain-info').textContent.includes('1.0×')`);
  await evaluate(`document.getElementById('limit').value='100';document.getElementById('limit').dispatchEvent(new Event('change'))`);
  if (!await evaluate(`document.getElementById('epoch').textContent==='1' && document.getElementById('observations').textContent==='2 observations'`)) throw new Error('Changing the budget reset training');
  await evaluate(`document.getElementById('loss-tab').click();document.getElementById('weight-view').value='delta';document.getElementById('weight-view').dispatchEvent(new Event('input'))`);
  for (const [width,height] of [[1280,720],[1024,768],[1920,1080],[1366,768]]) {
    await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
    await assertFits(`${width} × ${height} narrow`);
  }
  await writeFile('/tmp/grokking-dense-narrow.png',Buffer.from((await call<{data: string}>('Page.captureScreenshot',{captureBeyondViewport:true})).data,'base64'));
  await evaluate(`document.getElementById('train').click()`);
  await waitFor(`document.getElementById('status').textContent==='Training'`);
  await evaluate(`document.getElementById('train').click()`);
  await waitFor(`document.getElementById('status').textContent==='Paused'`);
  await evaluate(`document.getElementById('reset').click()`);
  await waitFor(`document.getElementById('status').textContent==='Ready' && document.getElementById('epoch').textContent==='0'`);
  await evaluate(`document.getElementById('backend').value='scalar';document.getElementById('backend').dispatchEvent(new Event('change'))`);
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
  await evaluate(`document.querySelector('[data-model-tab="dense-rotary-256"]').click()`);
  await waitFor(`document.getElementById('status').textContent==='Ready' && document.getElementById('epoch').textContent==='0'`);
  if (!await evaluate(`document.getElementById('model-detail').textContent.includes('256 channels') && document.getElementById('model-detail').textContent.includes('90,368') && document.getElementById('seed').value==='11' && document.getElementById('channel-select').max==='255' && document.getElementById('device').textContent.includes('WASM')`)) throw new Error('Incorrect wide defaults');
  await evaluate(`document.querySelector('[data-view-tab="weights"]').click()`);
  const initialMap = await evaluate(`document.getElementById('input-a').toDataURL()`);
  await evaluate(`document.querySelector('[data-view-tab="training"]').click();document.getElementById('step').click()`);
  await waitFor(`document.getElementById('epoch').textContent==='1' && document.getElementById('status').textContent==='Paused' && document.getElementById('observations').textContent==='2 observations'`);
  await evaluate(`document.querySelector('[data-view-tab="weights"]').click();new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))`);
  if (initialMap === await evaluate(`document.getElementById('input-a').toDataURL()`)) throw new Error('Shared embedding visualization did not update with training');
  if (!await evaluate(`document.getElementById('atlas-a-title').textContent==='Shared embedding E' && document.getElementById('atlas-b-title').textContent.includes('Dense matrix A') && document.getElementById('atlas-r-title').textContent==='Tied readout Eᵀ' && document.getElementById('atlas-a-size').textContent==='97 × 256' && document.getElementById('atlas-b-size').textContent==='256 × 256' && document.getElementById('atlas-r-size').textContent==='256 × 97'`)) throw new Error('Incorrect shared-E and rotated-E atlases');
  if (!await evaluate(`document.getElementById('atlas-b-size').textContent==='256 × 256'`)) throw new Error('Incorrect dense matrix atlas');
  await evaluate(`document.querySelector('[data-view-tab="channels"]').click();document.getElementById('channel-select').value='255';document.getElementById('channel-select').dispatchEvent(new Event('input'))`);
  await waitFor(`document.getElementById('mesh').getAttribute('aria-label').includes('140032 connections') && document.getElementById('mesh').getAttribute('aria-label').includes('90368 unique parameters')`);
  if (!await connectionBands(256)) throw new Error('Wide CPU drawing omitted a connection band');
  if (!await evaluate(`document.getElementById('mesh-edges').dataset.renderer==='CPU · Canvas 2D' && document.getElementById('mesh-edges').dataset.connections==='140032' && document.getElementById('mesh-edges').dataset.parameters==='90368'`)) throw new Error('Wide topology did not include every E role and dense edge');
  await evaluate(`document.getElementById('mesh').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight'}))`);
  await waitFor(`document.getElementById('channel-select').value==='0'`);
  await evaluate(`document.getElementById('mesh').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowLeft'}))`);
  await waitFor(`document.getElementById('channel-select').value==='255'`);
  await evaluate(`document.getElementById('mesh-channel').click()`);
  await waitFor(`document.getElementById('mesh').getAttribute('aria-label').includes('channel 255,') && document.getElementById('mesh').getAttribute('aria-label').includes('256 to 1 neurons, 256 learned edges, pre-RMSNorm, ReLU')`);
  if (!await evaluate(`document.getElementById('mesh-edges').hidden && document.getElementById('network-zoom').hidden`)) throw new Error('All-weights view leaked into the channel inspector');
  await evaluate(`document.getElementById('mesh-topology').click()`);
  await waitFor(`document.getElementById('mesh').getAttribute('aria-label').includes('140032 connections')`);
  for (const [width,height] of [[1366,768],[390,844],[320,568],[768,1024],[1366,600],[844,390]]) {
    await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width<650});
    for (const view of ['training','weights','channels']) {
      await evaluate(`document.querySelector('[data-view-tab="${view}"]').click()`);
      await assertFits(`${width} × ${height} wide ${view}`);
    }
  }
  const wide = optimizerProfile('dense-rotary-256');
  await evaluate(`document.getElementById('about').click()`);
  if (!await evaluate(`document.getElementById('about-dialog').open && document.getElementById('model-description').textContent.includes('Rotate90(E[b])') && document.getElementById('model-description').textContent.includes('One pre-norm residual block') && document.getElementById('reference-description').textContent.includes('seeds 11 / 17 / 23') && document.getElementById('reference-description').dataset.learningRate==='${wide.learningRate}' && document.getElementById('reference-description').dataset.weightDecay==='${wide.weightDecay}' && !document.getElementById('preset-nav').hidden`)) throw new Error('Incorrect model explanation');
  await evaluate(`document.getElementById('close-about').click()`);
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await writeFile('/tmp/grokking-dense-wide-mobile.png',Buffer.from((await call<{data: string}>('Page.captureScreenshot',{captureBeyondViewport:true})).data,'base64'));
  await evaluate(`document.querySelector('[data-model-tab="dense-rotary"]').click()`);
  const restored = await evaluate(`({epoch:document.getElementById('epoch').textContent,observations:document.getElementById('observations').textContent,loss:document.getElementById('train-loss').textContent})`);
  if (JSON.stringify(previous)!==JSON.stringify(restored)) throw new Error('Switching models changed the narrow experiment');
  if (!await evaluate(`document.getElementById('channel-select').max==='127'`)) throw new Error('Channel bounds did not follow width');
  // Optimizer presets are separate W128 experiments. W16 has no plateau preset.
  const slow = optimizerProfile('dense-rotary', 'slow'), plateau = optimizerProfile('dense-rotary', 'plateau'), w16 = optimizerProfile('dense-rotary-16');
  const preset = (name: string) => `document.querySelector('[data-preset-tab="${name}"]').getAttribute('aria-pressed')==='true'`;
  const offered = (name: string) => `!document.querySelector('[data-preset-tab="${name}"]').hidden`;
  const optimizer = (profile: {learningRate: number; weightDecay: number}) => `document.getElementById('reference-description').dataset.learningRate==='${profile.learningRate}' && document.getElementById('reference-description').dataset.weightDecay==='${profile.weightDecay}' && document.getElementById('model-optimizer').textContent==='lr=${profile.learningRate} · wd=${profile.weightDecay}'`;
  const epochs = (values: (number | null)[]) => values.map(epoch => epoch!.toLocaleString('en-US')).join(' / ');
  if (!await evaluate(`!document.getElementById('preset-nav').hidden && ${preset('fast')} && ${offered('plateau')} && ${offered('slow')}`)) throw new Error('W128 should offer fast, plateau and slow optimizer presets');
  await evaluate(`document.querySelector('[data-preset-tab="slow"]').click()`);
  await waitFor(`document.getElementById('status').textContent==='Ready' && document.getElementById('epoch').textContent==='0'`);
  if (!await evaluate(`${preset('slow')} && ${optimizer(slow)} && document.getElementById('reference-description').textContent.includes('${epochs(slow.nativeTest99Epochs)}')`)) throw new Error('The slow preset did not select its optimizer');
  for (const [width,height] of [[320,568],[844,390]]) {
    await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width<650});
    await assertFits(`${width} × ${height} slow preset`);
  }
  await evaluate(`document.querySelector('[data-preset-tab="plateau"]').click()`);
  await waitFor(`document.getElementById('status').textContent==='Ready' && document.getElementById('epoch').textContent==='0'`);
  if (!await evaluate(`${preset('plateau')} && ${optimizer(plateau)} && document.getElementById('reference-description').textContent.includes('Memorization plateau: ${epochs(plateau.nativePlateauEpochs!)} epochs.')`)) throw new Error('The plateau preset did not select its optimizer');
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate(`document.querySelector('[data-model-tab="dense-rotary-16"]').click()`);
  await waitFor(`document.getElementById('status').textContent==='Ready' && document.getElementById('model-detail').textContent.includes('16 channels')`);
  if (!await evaluate(`${preset('fast')} && !${offered('plateau')} && ${optimizer(w16)}`)) throw new Error('W16 has no plateau preset and should open its fast preset');
  await evaluate(`document.querySelector('[data-model-tab="dense-rotary"]').click()`);
  if (!await evaluate(`${preset('plateau')} && ${optimizer(plateau)} && document.getElementById('epoch').textContent==='0'`)) throw new Error('W128 should reopen the preferred plateau preset');
  await evaluate(`document.querySelector('[data-preset-tab="fast"]').click()`);
  const fastRestored = await evaluate(`({epoch:document.getElementById('epoch').textContent,observations:document.getElementById('observations').textContent,loss:document.getElementById('train-loss').textContent})`);
  if (JSON.stringify(previous)!==JSON.stringify(fastRestored)) throw new Error('Switching optimizer presets changed the fast experiment');
  await evaluate(`document.querySelector('[data-view-tab="channels"]').click();document.getElementById('channel-select').value='127';document.getElementById('channel-select').dispatchEvent(new Event('input'))`);
  await waitFor(`document.getElementById('mesh').getAttribute('aria-label').includes('channel 127,')`);
  await evaluate(`document.querySelector('[data-view-tab="training"]').click();document.getElementById('train').click()`);
  await waitFor(`document.getElementById('status').textContent==='Training'`);
  await evaluate(`document.querySelector('[data-model-tab="dense-rotary-256"]').click()`);
  if (!await evaluate(`document.getElementById('epoch').textContent==='1' && document.getElementById('status').textContent==='Paused' && document.getElementById('channel-select').value==='255'`)) throw new Error('Wide state was not restored');
  await new Promise(resolve=>setTimeout(resolve,250));
  await evaluate(`document.querySelector('[data-model-tab="dense-rotary"]').click()`);
  await waitFor(`document.getElementById('status').textContent==='Paused' || document.getElementById('status').textContent==='Epoch limit reached'`);
  await call('Emulation.setDeviceMetricsOverride',{width:1366,height:768,deviceScaleFactor:1,mobile:false});
  await evaluate(`document.querySelector('[data-model-tab="dense-rotary-256"]').click();document.querySelector('[data-view-tab="training"]').click();document.getElementById('loss-tab').click()`);
  await writeFile('/tmp/grokking-dense-wide.png',Buffer.from((await call<{data: string}>('Page.captureScreenshot',{captureBeyondViewport:true})).data,'base64'));
  if (errors.length) throw new Error(`Browser exceptions: ${JSON.stringify(errors)}`);
  console.log('PASS: W128 and W256, the W128 plateau and slow presets and all three views fit, CPU/WASM training with GPU APIs forbidden, off-thread software drawing of all connections including three E roles, fixed-scale live weights and changes, highlighting, zoom, inspection, live atlases, pause/reset, independent states and model information.');
} finally { client.close(); }
