import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {CDP} from './cdp.ts';

const client = await CDP.connect(Number(process.env.GROKKING_CDP_PORT || 9224));
const origin = process.env.GROKKING_PREVIEW_ORIGIN || 'http://127.0.0.1:8080';
const evaluate = client.evaluate.bind(client), wait = client.wait.bind(client);
async function click(selector: string) {await evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);}
async function values(width: string, lr: string, wd: string) {
  await evaluate(`(() => {for(const [id,value] of ${JSON.stringify([['custom-width', width], ['custom-lr', lr], ['custom-wd', wd]])}){const input=document.getElementById(id);input.value=value;input.dispatchEvent(new Event('input'));}})()`);
}
async function apply(width: number, lr: number, wd: number) {
  await click('[data-view-tab="settings"]'); await values(String(width), String(lr), String(wd));
  await evaluate(`document.getElementById('settings-form').requestSubmit()`);
  await wait(`document.getElementById('status').textContent==='Ready' && document.getElementById('model-detail').textContent.startsWith('${width} channels')`);
  assert.ok(await evaluate(`document.body.dataset.view==='training'&&document.getElementById('custom-tab').getAttribute('aria-pressed')==='true'&&document.getElementById('preset-nav').hidden&&document.getElementById('model-optimizer').textContent==='lr=${lr} · wd=${wd}'`));
}
async function fits(label: string) {
  await evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
  const issues = await evaluate<string[]>(`(() => {const issues=[];if(document.documentElement.scrollWidth>innerWidth||document.documentElement.scrollHeight>innerHeight)issues.push('page scrolling');for(const element of document.querySelectorAll('.workspace-nav button,#settings-panel,#settings-panel input,#settings-panel button')){if(!element.getClientRects().length)continue;const r=element.getBoundingClientRect();if(r.left<0||r.top<0||r.right>innerWidth+.5||r.bottom>innerHeight+.5)issues.push(element.id||element.textContent);if(element.id==='settings-panel'&&(element.scrollHeight>element.clientHeight+1||element.scrollWidth>element.clientWidth+1))issues.push('settings content clipped');}return issues})()`);
  assert.deepEqual(issues, [], label);
}
try {
  await client.call('Runtime.enable'); await client.call('Page.enable');
  await client.call('Emulation.setDeviceMetricsOverride', {width: 1366, height: 768, deviceScaleFactor: 1, mobile: false});
  await client.call('Page.navigate', {url: origin + '?lang=en'});
  await wait(`document.getElementById('task-dialog')?.open&&document.getElementById('status')?.textContent==='Ready'`);
  const route = [...['why', 'epoch', 'paper', 'chart', 'grokking', 'findings'].map(name => 'intro/' + name), 'task', 'solution', 'examples', ...['diagram', 'types', 'equations'].map(name => 'model/' + name)];
  const current = `(() => {const page=document.querySelector('[data-task-tab][aria-selected="true"]').dataset.taskTab;return page==='intro'?page+'/'+document.querySelector('[data-intro-tab][aria-selected="true"]').dataset.introTab:page==='model'?page+'/'+document.querySelector('[data-model-explanation][aria-selected="true"]').dataset.modelExplanation:page})()`;
  assert.ok(await evaluate(`document.getElementById('task-back').disabled`));
  for (const [index, page] of route.entries()) {
    if (index) await click('#task-next');
    assert.equal(await evaluate(current), page);
  }
  assert.ok(await evaluate(`document.getElementById('task-next').disabled`));
  for (const page of route.slice(0, -1).reverse()) {await click('#task-back'); assert.equal(await evaluate(current), page);}
  assert.ok(await evaluate(`document.getElementById('task-back').disabled`));
  await click('[data-intro-tab="findings"]'); await click('#task-next'); assert.equal(await evaluate(current), 'task');
  await click('#task-back'); assert.equal(await evaluate(current), 'intro/findings');
  await click('#task-open');
  await evaluate(`document.getElementById('backend').value='simd';document.getElementById('backend').dispatchEvent(new Event('change'))`);
  await wait(`document.getElementById('status').textContent==='Ready'`);
  await click('#step'); await wait(`document.getElementById('epoch').textContent==='1'&&document.getElementById('status').textContent==='Paused'`);
  const nativeLoss = await evaluate(`document.getElementById('train-loss').textContent`);
  await apply(48, .0025, 0);
  const initialLoss = await evaluate(`document.getElementById('train-loss').textContent`);
  const source = await evaluate<string>(`(async()=>{let source;const create=URL.createObjectURL,click=HTMLAnchorElement.prototype.click;URL.createObjectURL=blob=>{source=blob.text();return create(blob)};HTMLAnchorElement.prototype.click=function(){};try{document.getElementById('export-pytorch').click();return await source}finally{URL.createObjectURL=create;HTMLAnchorElement.prototype.click=click}})()`);
  for (const line of ['DEFAULT_WIDTH = 48', 'DEFAULT_LEARNING_RATE = 0.0025', 'DEFAULT_WEIGHT_DECAY = 0']) assert.ok(source.includes(line + '\n'));
  await click('#step'); await wait(`document.getElementById('epoch').textContent==='1'&&document.getElementById('status').textContent==='Paused'`);
  const customLoss = await evaluate(`document.getElementById('train-loss').textContent`);
  assert.notEqual(customLoss, initialLoss);
  await click('[data-model-tab="dense-rotary"]');
  assert.equal(await evaluate(`document.getElementById('train-loss').textContent`), nativeLoss);
  assert.ok(await evaluate(`document.getElementById('epoch').textContent==='1'&&document.getElementById('model-optimizer').textContent==='lr=0.01 · wd=1'`));
  await click('#custom-tab'); assert.equal(await evaluate(`document.getElementById('train-loss').textContent`), customLoss);
  await click('#reset'); await wait(`document.getElementById('epoch').textContent==='0'&&document.getElementById('status').textContent==='Ready'`);
  assert.equal(await evaluate(`document.getElementById('train-loss').textContent`), initialLoss);
  assert.equal(await evaluate(`document.getElementById('model-optimizer').textContent`), 'lr=0.0025 · wd=0');
  await apply(18, .003, .2);
  assert.ok(await evaluate(`document.getElementById('device').textContent.includes('scalar')`));
  await click('#step'); await wait(`document.getElementById('epoch').textContent==='1'&&document.getElementById('status').textContent==='Paused'`);
  await click('[data-view-tab="weights"]'); await wait(`document.getElementById('atlas-b-size').textContent==='18 × 18'`);
  await click('[data-view-tab="channels"]');
  await wait(`document.getElementById('mesh-edges').dataset.parameters==='2070'&&document.getElementById('mesh-edges').dataset.frame===document.getElementById('mesh-edges').dataset.request`);
  await click('[data-view-tab="settings"]');
  for (const invalid of [['17', '.003', '.2'], ['2050', '.003', '.2'], ['18', '0', '.2'], ['18', '.003', '-1'], ['', '.003', '.2']]) {
    await values(...invalid as [string, string, string]);
    assert.equal(await evaluate(`document.getElementById('settings-form').checkValidity()`), false);
    await evaluate(`document.getElementById('settings-form').requestSubmit()`);
    assert.ok(await evaluate(`document.body.dataset.view==='settings'&&document.getElementById('epoch').textContent==='1'`));
  }
  await values('18', '.003', '.2');
  for (const language of ['en', 'ru', 'de', 'ja', 'es', 'fr']) {
    await evaluate(`document.getElementById('language').value='${language}';document.getElementById('language').dispatchEvent(new Event('change'))`);
    await wait(`document.documentElement.lang==='${language}'`);
    for (const [width, height] of [[1366, 768], [700, 800], [390, 844], [320, 568], [844, 390]]) {
      await client.call('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: 1, mobile: width < 650});
      await fits(`${language} ${width}×${height}`);
      if (language === 'ru' && width === 1366) await writeFile('/tmp/grokking-settings-ru.png', Buffer.from((await client.call<{data: string}>('Page.captureScreenshot')).data, 'base64'));
      if (language === 'ru' && width === 320) await writeFile('/tmp/grokking-settings-mobile-ru.png', Buffer.from((await client.call<{data: string}>('Page.captureScreenshot')).data, 'base64'));
    }
  }
  assert.deepEqual(client.errors, []);
  console.log('Introduction: all 12 pages forward and backward. Custom W48 SIMD and W18 scalar: training, reset, independent presets, export and validation passed. Settings: 30 localized layouts passed.');
} finally {client.close();}
