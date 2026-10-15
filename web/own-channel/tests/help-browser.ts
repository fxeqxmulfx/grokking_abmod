import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {CDP} from './cdp.ts';
import {HELP_TOPICS} from '../dist/help-content.js';
import type {HelpId, HelpTopic} from '../dist/help-content.js';
import {languages, rows} from '../scripts/translations.ts';

const client = await CDP.connect(Number(process.env.GROKKING_CDP_PORT || 9224));
const origin = (process.env.GROKKING_PREVIEW_ORIGIN || 'http://127.0.0.1:8080').replace(/\/$/, '');
const evaluate = client.evaluate.bind(client), wait = client.wait.bind(client);
const click = (selector: string) => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
const paint = () => evaluate('document.fonts.ready.then(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))))');
const snapshot = `JSON.stringify([...document.querySelectorAll('main input,main select')].map(input=>[input.id,input.value,input.checked]),null,0)`;
let opened = 0;
async function fits(label: string) {
  await paint();
  const issues = await evaluate<string[]>(`(() => {
    const issues=[],dialog=document.getElementById('help-dialog'),content=document.getElementById('help-content');
    for(const element of [dialog,...dialog.querySelectorAll('h2,button')]){
      if(element.hidden)continue;const r=element.getBoundingClientRect();
      if(r.left<0||r.right>innerWidth+.5||r.top<0||r.bottom>innerHeight+.5)issues.push(element.id+' leaves viewport');
      if(element.scrollWidth>element.clientWidth+1)issues.push(element.id+' requires horizontal scrolling');
    }
    if(content.scrollWidth>content.clientWidth+1)issues.push('help text requires horizontal scrolling');
    if(Number.parseFloat(getComputedStyle(content).fontSize)<14)issues.push('help text is too small');
    if(document.documentElement.scrollWidth>innerWidth)issues.push('page requires horizontal scrolling');
    return issues;
  })()`);
  assert.deepEqual(issues, [], label);
}
try {
  await client.call('Page.enable'); await client.call('Runtime.enable'); await client.call('Page.bringToFront');
  await client.call('Emulation.setDeviceMetricsOverride', {width: 1366, height: 768, deviceScaleFactor: 1, mobile: false});
  await client.call('Page.navigate', {url: origin + '/?lang=en'});
  await wait(`document.getElementById('task-dialog')?.open&&document.getElementById('status')?.textContent==='Ready'`);
  await click('#task-open');
  assert.equal(await evaluate(`document.querySelectorAll('label button[data-help]').length`), 0, 'Help must not be nested inside input labels.');
  for (const [languageIndex, language] of languages.entries()) {
    const dictionary = Object.fromEntries(rows.map(row => [row[0], row[languageIndex]]));
    await evaluate(`document.getElementById('language').value='${language}';document.getElementById('language').dispatchEvent(new Event('change'))`);
    await wait(`document.documentElement.lang==='${language}'`);
    for (const [width, height] of [[1366, 768], [320, 568], [844, 390]]) {
      await client.call('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: width === 320 ? 3 : 1, mobile: width === 320});
      const covered = new Set<HelpId>();
      for (const view of ['training', 'weights', 'channels', 'settings']) {
        await click(`[data-view-tab="${view}"]`); await paint();
        const buttons = await evaluate<{index: number; topic: HelpId}[]>(`Array.from(document.querySelectorAll('button[data-help]'),(button,index)=>({button,index})).filter(({button})=>button.getBoundingClientRect().width>0).map(({button,index})=>({index,topic:button.dataset.help}))`);
        for (const {index, topic: id} of buttons) {
          if (covered.has(id)) continue;
          const topic: HelpTopic = HELP_TOPICS[id], before = await evaluate(snapshot);
          await evaluate(`window.helpOpener=document.querySelectorAll('button[data-help]')[${index}];window.helpOpener.focus();window.helpOpener.click()`);
          assert.equal(await evaluate(`document.getElementById('help-dialog').open`), true);
          assert.equal(await evaluate(`document.getElementById('help-title').textContent`), dictionary[topic.title]);
          assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('#help-content>p'),p=>p.textContent)`), [...topic.paragraphs.map(text => dictionary[text]), ...(topic.formula ? [topic.formula] : [])]);
          assert.equal(await evaluate(`document.getElementById('help-guide').hidden`), !topic.guide);
          await fits(`${language} ${width}×${height} ${id}`);
          await click('#help-done');
          assert.equal(await evaluate(`document.activeElement===window.helpOpener`), true, 'Closing help restores keyboard focus.');
          assert.equal(await evaluate(snapshot), before, 'Opening help must not change settings or checkboxes.');
          covered.add(id); opened++;
        }
      }
      assert.deepEqual(covered, new Set(Object.keys(HELP_TOPICS)), 'Every topic must be reachable in all layouts and languages.');
    }
  }
  // Native keyboard activation and Escape must work as well as a pointer click.
  await click('[data-view-tab="settings"]');
  await evaluate(`document.querySelector('[data-help="lr"]').focus()`);
  await client.call('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, text: '\r', unmodifiedText: '\r'});
  await client.call('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13});
  assert.equal(await evaluate(`document.getElementById('help-dialog').open`), true);
  await client.call('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  await client.call('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  await wait(`!document.getElementById('help-dialog').open`);
  assert.equal(await evaluate(`document.activeElement===document.querySelector('[data-help="lr"]')`), true);
  // Use real touch input to verify the target and backdrop on a narrow phone.
  await client.call('Emulation.setDeviceMetricsOverride', {width: 320, height: 568, deviceScaleFactor: 3, mobile: true});
  await client.call('Emulation.setTouchEmulationEnabled', {enabled: true});
  const target = await evaluate<{x: number; y: number}>(`(() => {const button=document.querySelector('[data-help="wd"]');button.scrollIntoView({block:'center'});const r=button.getBoundingClientRect();return{x:r.left+r.width/2,y:r.top+r.height/2}})()`);
  await client.call('Input.dispatchTouchEvent', {type: 'touchStart', touchPoints: [target]});
  await client.call('Input.dispatchTouchEvent', {type: 'touchEnd', touchPoints: []});
  await wait(`document.getElementById('help-dialog').open`);
  await client.call('Input.dispatchTouchEvent', {type: 'touchStart', touchPoints: [{x: 4, y: 4}]});
  await client.call('Input.dispatchTouchEvent', {type: 'touchEnd', touchPoints: []});
  await wait(`!document.getElementById('help-dialog').open`);
  assert.equal(await evaluate(`document.activeElement===document.querySelector('[data-help="wd"]')`), true);
  for (const [id, guide] of [['curves', 'chart'], ['weights', 'weights'], ['contributions', 'probe'], ['network', 'model'], ['probe', 'task']]) {
    await evaluate(`document.querySelector('[data-help="${id}"]').click()`); await click('#help-guide');
    assert.equal(await evaluate(`document.getElementById('task-dialog').open&&!document.getElementById('${guide==='model'||guide==='task'?'task-page-':'intro-page-'}${guide}').hidden`), true);
    await click('#task-open');
  }
  assert.equal(await evaluate(`document.getElementById('epoch').textContent`), '0', 'Help must not start or reset training.');
  assert.deepEqual(client.errors, []);
  await writeFile('/tmp/grokking-help-verification.json', JSON.stringify({origin,topics:Object.keys(HELP_TOPICS).length,opened,languages,keyboard:true,touch:true,backdrop:true,focusRestoration:true,settingsPreserved:true,errors:client.errors},null,2)+'\n');
  console.log(`PASS: ${Object.keys(HELP_TOPICS).length} help topics, ${opened} localized dialogs, desktop/phone/landscape layouts, readable text, focus, Enter/Escape, touch, backdrop, guide links and unchanged settings.`);
} finally {client.close();}
