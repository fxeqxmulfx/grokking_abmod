import {CDP} from './cdp.ts';
import {writeFile} from 'node:fs/promises';
const client = await CDP.connect(Number(process.env.GROKKING_CDP_PORT || 9224));
const origin = process.env.GROKKING_PREVIEW_ORIGIN || 'http://127.0.0.1:8080';
const evaluate = client.evaluate.bind(client);
const paint = async () => {
  await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
  await client.wait(`(() => {const d=document.getElementById('mesh-edges').dataset;return d.frame===d.request})()`);
  await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
};
try {
  await client.call('Page.enable'); await client.call('Runtime.enable');
  await client.call('Emulation.setDeviceMetricsOverride', {width: 1366, height: 768, deviceScaleFactor: 1, mobile: false});
  await client.call('Page.navigate', {url: origin + '?lang=en'});
  await client.wait(`document.getElementById('status')?.textContent==='Ready' && document.getElementById('task-dialog').open`);
  await evaluate(`document.getElementById('task-open').click();document.querySelector('[data-model-tab="dense-rotary-256"]').click();document.querySelector('[data-view-tab="channels"]').click()`);
  await client.wait(`document.getElementById('status')?.textContent==='Ready'`); await paint();
  const initial = await evaluate(`document.getElementById('mesh-edges').toDataURL()`);
  const builds = await evaluate(`document.getElementById('mesh-edges').dataset.classifications`);
  await evaluate(`document.getElementById('zoom-in').click();document.getElementById('zoom-in').click()`); await paint();
  const rect = await evaluate<{x: number; y: number}>(`(()=>{const r=document.getElementById('mesh').getBoundingClientRect();return{x:r.left+r.width/2,y:r.top+r.height/2}})()`);
  await client.call('Input.dispatchMouseEvent', {type: 'mousePressed', x: rect.x, y: rect.y, button: 'left', buttons: 1, clickCount: 1});
  await client.call('Input.dispatchMouseEvent', {type: 'mouseMoved', x: rect.x + 60, y: rect.y, button: 'left', buttons: 1});
  await client.call('Input.dispatchMouseEvent', {type: 'mouseReleased', x: rect.x + 60, y: rect.y, button: 'left', buttons: 0, clickCount: 1});
  await paint();
  if (initial === await evaluate(`document.getElementById('mesh-edges').toDataURL()`)) throw new Error('Pointer pan did not move the graph.');
  if (builds !== await evaluate(`document.getElementById('mesh-edges').dataset.classifications`)) throw new Error('Pan/zoom rebuilt unchanged classifications.');
  await evaluate(`document.getElementById('zoom-reset').click()`); await paint();
  if (initial !== await evaluate(`document.getElementById('mesh-edges').toDataURL()`)) throw new Error('Fit did not restore the graph.');
  const checks = await evaluate(`(async () => {
    const {CPUScene,topologyLayout}=await import('/topology-scene.js'),{WeightTopology}=await import('/topology.js'),{initialModelWeights}=await import('/models.js');
    const results=[];
    for(const width of [128,256]){
      const canvas=document.createElement('canvas');canvas.width=1250;canvas.height=520;
      const ctx=canvas.getContext('2d',{willReadFrequently:true}),layout=topologyLayout(1250,520,width),weights=initialModelWeights(11,width===128?'dense-rotary':'dense-rotary-256'),scene=await CPUScene.create(),view={zoom:1,x:0,y:0};
      const fresh=[],pan=[];let current=weights;
      for(let i=0;i<7;i++){current=weights.slice();const start=performance.now();scene.draw(ctx,layout,current,32,view,-1,{initial:weights});fresh.push(performance.now()-start)}
      const classifications=scene.classifications;
      for(let i=0;i<7;i++){const start=performance.now();scene.draw(ctx,layout,current,32,{zoom:1+i*.1,x:i,y:0},-1,{initial:weights});pan.push(performance.now()-start)}
      if(scene.classifications!==classifications)throw new Error('Pan rebuilt the classification cache');
      scene.draw(ctx,layout,current,32,view,-1,{initial:weights});
      const backgrounds=scene.backgroundPaints,start=performance.now();scene.draw(ctx,layout,current,33,view,-1,{initial:weights,highlight:true});const highlightMs=performance.now()-start;
      if(scene.backgroundPaints!==backgrounds||scene.classifications!==classifications)throw new Error('Highlight rebuilt the background');
      const bandVisibility=()=>{const data=ctx.getImageData(0,0,1250,520).data;return layout.rows.slice(0,3).map((y,i)=>{let n=0;const row=Math.floor((y+layout.rows[i+1])/2);for(let x=0;x<1250;x++)if(data[(row*1250+x)*4+3]>0)n++;return n})};
      if(!bandVisibility().every(n=>n>1250/4))throw new Error('Raw graph omitted connections');
      scene.draw(ctx,layout,weights,32,view,-1,{initial:weights,mode:'delta'});
      if(bandVisibility().some(n=>n>0))throw new Error('Initial changes should be transparent');
      const changed=weights.map((v,i)=>v+(i<97*width?.002:.01));scene.draw(ctx,layout,changed,32,view,-1,{initial:weights,mode:'delta'});
      if(!bandVisibility().every(n=>n>1250/4))throw new Error('Weak changes disappeared');
      // Explicitly exercise the main-thread fallback with the real wrapper.
      const fallbackCanvas=document.createElement('canvas');Object.defineProperty(fallbackCanvas,'transferControlToOffscreen',{value:undefined});
      let fallback;await new Promise(resolve=>{fallback=new WeightTopology(fallbackCanvas,resolve);fallback.draw(ctx,layout,weights,32,view,-1,{initial:weights})});fallback.draw(ctx,layout,weights,32,view,-1,{initial:weights});
      if(fallbackCanvas.dataset.thread!=='main'||Number(fallbackCanvas.dataset.connections)!==3*97*width+width*width)throw new Error('CPU fallback failed');fallback.destroy();
      const median=values=>values.slice(2).sort((a,b)=>a-b)[Math.floor((values.length-2)/2)];
      results.push({width,connections:3*97*width+width*width,freshMs:fresh,freshMedianMs:median(fresh),panMs:pan,panMedianMs:median(pan),highlightMs,classificationReuse:true,backgroundReuse:true,allBandsVisible:true,weakChangesVisible:true,fallbackWorks:true});
    }
    return{viewport:[1250,520],renderer:'CPU Canvas 2D',results};
  })()`);
  if (client.errors.length) throw new Error(JSON.stringify(client.errors));
  await writeFile('/tmp/grokking-cpu-topology.json', JSON.stringify(checks, null, 2) + '\n');
  console.log(JSON.stringify(checks));
  console.log('PASS: all connections, raw/change, software fallback, cached classifications/background, real pointer pan and Fit.');
} finally {client.close();}
