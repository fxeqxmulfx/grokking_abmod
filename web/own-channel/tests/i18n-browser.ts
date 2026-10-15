import {CDP} from './cdp.ts';
import {root} from '../project.ts';
import {writeFile} from 'node:fs/promises';
import {languages, rows} from '../scripts/translations.ts';

const port = Number(process.env.GROKKING_CDP_PORT || 9224), origin = process.env.GROKKING_PREVIEW_ORIGIN || 'http://127.0.0.1:8080';
const client = await CDP.connect(port);
const call = client.call.bind(client), evaluate = client.evaluate.bind(client), errors = client.errors;
const waitFor = client.wait.bind(client), wait = waitFor;
let layouts = 0;
async function frames() { await evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))'); }
async function assertFits(label: string, modal = false) {
  await frames();
  const issues = await evaluate<string[]>(`(() => {
    const issues=[],root=document.documentElement;
    if(root.scrollWidth>innerWidth||root.scrollHeight>innerHeight)issues.push('page scrolling');
    const selector=${JSON.stringify(modal ? '#task-dialog, #task-dialog .task-heading, #task-dialog .task-tabs, #task-dialog .task-pages, #task-dialog .task-page, #task-dialog .task-footer, #task-dialog p, #task-dialog code, #task-dialog button, #task-dialog select, #task-dialog img, #task-dialog table' : '.topbar, main .card, main button, main input, main select, main canvas, .header-tools')};
    for(const element of document.querySelectorAll(selector)) {
      if(element.closest('[hidden]'))continue;
      const rect=element.getBoundingClientRect();if(!rect.width||!rect.height)continue;
      const name=element.id||element.className||element.tagName;
      if(rect.left<-.5||rect.top<-.5||rect.right>innerWidth+.5||rect.bottom>innerHeight+.5)issues.push(name+' leaves viewport');
      if(element.matches('.card,.topbar,.header-tools,.task-dialog,.task-page,.task-pages,table')&&(element.scrollWidth>element.clientWidth+1||element.scrollHeight>element.clientHeight+1))issues.push(name+' clips content');
      if(${modal}&&!element.matches('#task-dialog')){const panel=document.getElementById('task-dialog').getBoundingClientRect();if(rect.left<panel.left||rect.right>panel.right||rect.top<panel.top||rect.bottom>panel.bottom)issues.push(name+' leaves dialog');}
      if(element.tagName==='IMG'&&(!element.complete||!element.naturalWidth))issues.push('SVG missing');
    }
    // Header text and navigation buttons must not overlap one another.
    const controls=${modal}?[]:[...document.querySelectorAll('h1, .model-task code, .device, .header-tools > *, .workspace-nav button')].filter(element=>!element.closest('[hidden]')&&element.getClientRects().length);
    const boxes=controls.map(element=>element.getBoundingClientRect());
    boxes.forEach((box,i)=>{for(let j=0;j<i;j++)if(box.left<boxes[j].right-.5&&boxes[j].left<box.right-.5&&box.top<boxes[j].bottom-.5&&boxes[j].top<box.bottom-.5)issues.push((controls[i].id||controls[i].textContent)+' overlaps '+(controls[j].id||controls[j].textContent));});
    return issues;
  })()`);
  if (issues.length) throw new Error(`${label}: ${issues.join('; ')}`);
  layouts++;
}
async function choose(language: string) {
  await evaluate(`document.getElementById('language').value=${JSON.stringify(language)};document.getElementById('language').dispatchEvent(new Event('change'))`);
  await wait(`document.documentElement.lang===${JSON.stringify(language)}`); await frames();
}
const expected = (language: string) => Object.fromEntries(rows.map(row=>[row[0],row[languages.findIndex(value => value === language)]]));
try {
  await call('Runtime.enable');
  await call('Page.enable');
  const injected = await call('Page.addScriptToEvaluateOnNewDocument', {source:`Object.defineProperty(navigator,'languages',{configurable:true,get:()=>['es-MX','en-US']});try{localStorage.removeItem('grokking.locale')}catch{}`});
  await call('Emulation.setDeviceMetricsOverride',{width:1366,height:768,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:origin});
  await wait(`document.documentElement?.lang==='es'&&document.getElementById('task-dialog')?.open&&[...document.querySelectorAll('[data-task-diagram]')].every(image=>image.complete&&image.naturalWidth>0)`);
  await call('Page.removeScriptToEvaluateOnNewDocument',{identifier:injected.identifier});
  if (!await evaluate(`document.getElementById('task-title').textContent===${JSON.stringify(expected('es')['Division modulo 97'])}&&document.getElementById('language').value==='auto'`)) throw new Error('Browser language was not selected automatically.');
  for (const language of languages) {
    await choose(language);
    if (!await evaluate(`document.getElementById('task-language').value===${JSON.stringify(language)}&&document.getElementById('task-title').textContent===${JSON.stringify(expected(language)['Division modulo 97'])}`)) throw new Error(`Startup dialog is not translated: ${language}`);
    for (const [width,height] of [[1366,768],[390,844],[320,568],[844,390]]) {
      await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width<650});
      for (const page of ['intro','task','solution','examples','model']) {
        await evaluate(`document.querySelector('[data-task-tab="${page}"]').click()`);
        await assertFits(`${language} ${width}×${height} intro ${page}`,true);
        if (page === 'intro') for (const detail of ['why','epoch','paper','chart','grokking','findings']) {
          await evaluate(`document.querySelector('[data-intro-tab="${detail}"]').click()`);
          await assertFits(`${language} ${width}×${height} introduction ${detail}`,true);
        }
        if (page === 'model') {
          for (const view of ['types','equations']) {
            await evaluate(`document.querySelector('[data-model-explanation="${view}"]').click()`);
            await assertFits(`${language} ${width}×${height} model ${view}`,true);
          }
          await evaluate(`document.querySelector('[data-model-explanation="diagram"]').click()`);
        }
      }
    }
  }
  await call('Emulation.setDeviceMetricsOverride',{width:1366,height:768,deviceScaleFactor:1,mobile:false});
  await choose('ru'); await evaluate(`document.querySelector('[data-task-tab="task"]').click()`);
  await writeFile('/tmp/grokking-intro-ru.png',Buffer.from((await call<{data: string}>('Page.captureScreenshot')).data,'base64'));
  await evaluate(`document.getElementById('task-open').click();document.getElementById('backend').value='simd';document.getElementById('backend').dispatchEvent(new Event('change'))`);
  await wait(`document.getElementById('status').textContent===${JSON.stringify(expected('ru').Ready)}`);
  await evaluate(`document.getElementById('step').click()`);
  await wait(`document.getElementById('status').textContent===${JSON.stringify(expected('ru').Paused)}`);
  const snapshot = `({epoch:document.getElementById('epoch').textContent,observations:document.getElementById('observations').textContent,loss:document.getElementById('train-loss').textContent})`;
  const baseline = await evaluate(snapshot);
  for (const language of languages) {
    await choose(language);
    if (!await evaluate(`document.getElementById('reset').textContent===${JSON.stringify(expected(language).Reset)}&&document.getElementById('status').textContent===${JSON.stringify(expected(language).Paused)}&&document.getElementById('task-language').value===${JSON.stringify(language)}`)) throw new Error(`Live UI is not translated: ${language}`);
    // Locale-specific number formatting differs; a reset would change epoch 1.
    if (!await evaluate(`document.getElementById('epoch').textContent==='1'`)) throw new Error(`Language switch reset training: ${language}`);
    for (const [width,height] of [[1366,768],[700,800],[390,844],[320,568],[844,390]]) {
      await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width<650});
      for(const view of ['training','weights','channels','settings']) {
        await evaluate(`document.querySelector('[data-view-tab="${view}"]').click()`);
        await assertFits(`${language} ${width}×${height} ${view}`);
      }
    }
  }
  await choose('ja');
  await call('Emulation.setDeviceMetricsOverride',{width:1366,height:768,deviceScaleFactor:1,mobile:false});
  await writeFile('/tmp/grokking-training-ja.png',Buffer.from((await call<{data: string}>('Page.captureScreenshot')).data,'base64'));
  await call('Page.navigate',{url:origin});
  await wait(`document.documentElement?.lang==='ja'&&document.getElementById('task-dialog')?.open`);
  if(!await evaluate(`document.getElementById('language').value==='ja'`))throw new Error('Manual language was not persisted.');
  await evaluate(`document.getElementById('task-language').value='auto';document.getElementById('task-language').dispatchEvent(new Event('change'))`);
  await wait(`document.documentElement.lang==='en'&&document.getElementById('language').value==='auto'`);
  await evaluate(`Object.defineProperty(navigator,'languages',{configurable:true,get:()=>['ru-KZ','en-US']});window.dispatchEvent(new Event('languagechange'))`);
  await wait(`document.documentElement.lang==='ru'`);
  await evaluate(`document.getElementById('task-open').click();document.getElementById('task-help').click()`);
  if(!await evaluate(`document.getElementById('task-dialog').open&&document.querySelectorAll('[data-task-tab]').length===5&&!document.getElementById('task-dialog').textContent.includes('EOS')`))throw new Error('Reopening the dialog or reference-free content failed.');
  await evaluate(`document.getElementById('task-open').click();document.getElementById('chart-help').click()`);
  if(!await evaluate(`document.getElementById('task-dialog').open&&!document.getElementById('task-page-intro').hidden&&!document.getElementById('intro-page-chart').hidden&&document.getElementById('intro-tab-chart').getAttribute('aria-selected')==='true'`))throw new Error('Chart guide did not open the shading explanation.');
  if(errors.length)throw new Error(JSON.stringify(errors));
  await writeFile('/tmp/grokking-i18n-verification.json',JSON.stringify({languages,layouts,automaticRegionalSelection:true,persistedManualChoice:true,browserLanguageChange:true,languageSwitchPreservesTraining:true,startupAndReopen:true,localRenderedSVG:true,errors},null,2)+'\n');
  console.log(`PASS: ${languages.length} languages, ${layouts} layout checks, automatic selection, persisted manual choice, live translations preserve training, real local SVGs, startup and reopen, no reference section.`);
} finally { client.close(); }
