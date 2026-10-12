import {mkdir, writeFile, readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';

const model = process.argv[2], directory = process.argv[3];
if (!['concat','residual','neighbor','neighbor-concat','neighbor-rotary','neighbor-rotary-wide'].includes(model) || !directory) throw new Error('Usage: node scripts/measure-browser.mjs <model> <output-directory>');
const port = process.env.GROKKING_CDP_PORT || 9224, origin = process.env.GROKKING_PREVIEW_ORIGIN || 'http://127.0.0.1:8080';
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(), target = targets.find(t => t.type === 'page');
if (!target) throw new Error('Start isolated test Chrome first.');
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject) => { socket.addEventListener('open',resolve,{once:true}); socket.addEventListener('error',reject,{once:true}); });
try {
  const result = await new Promise((resolve,reject) => {
    socket.addEventListener('message',({data}) => {
      const message = JSON.parse(data);
      if (message.id !== 1) return;
      if (message.error || message.result.exceptionDetails) reject(new Error(JSON.stringify(message.error || message.result.exceptionDetails)));
      else resolve(message.result.result.value);
    });
    socket.send(JSON.stringify({id:1,method:'Runtime.evaluate',params:{expression:`import(${JSON.stringify(origin+'/tests/measure-gpu.mjs')}).then(m=>m.measureGPU(${JSON.stringify(model)}))`,awaitPromise:true,returnByValue:true}}));
  });
  const history = result.history; delete result.history;
  result.node = process.version; result.sourceSha256 = {};
  for (const name of [`${model}.wgsl`,'gpu.mjs','models.mjs','model.mjs']) result.sourceSha256[name] = createHash('sha256').update(await readFile(new URL(`../dist/${name}`,import.meta.url))).digest('hex');
  await mkdir(directory,{recursive:true});
  await writeFile(`${directory}/run.json`,JSON.stringify(result,null,2)+'\n');
  await writeFile(`${directory}/history.json`,JSON.stringify(history,null,2)+'\n');
  console.log(JSON.stringify({...result,final:history.at(-1)}));
} finally { socket.close(); }
