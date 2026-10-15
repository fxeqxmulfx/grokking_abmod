// Sequential CUDA training avoids contention and retains every trial history.
// Usage: node scripts/optimizer-grid.mjs [output] [plan]
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdir, readFile, writeFile, open, stat, rename} from 'node:fs/promises';
import {resolve, join} from 'node:path';
import {eligible,rank,screenBudget,robustBound} from './optimizer-grid-policy.mjs';

const root = resolve(new URL('..', import.meta.url).pathname);
const directory = resolve(process.argv[2] ?? join(root,'target/optimizer-grid/search'));
const plan = JSON.parse(await readFile(resolve(process.argv[3] ?? join(root,'experiments/div97-dense-optimizer-grid.json')),'utf8'));
const original = join(root,'target/grok-div97-dense-embedding-normal-width/bin/mesh-gpt');
const readyPath = join(root,'target/optimizer-grid/cuda-ready.json');
await mkdir(directory,{recursive:true});
try {
  const previousPlan=JSON.parse(await readFile(join(directory,'plan.json'),'utf8'));
  const fixed=({learningRates,weightDecays,screenRacing,worstSeedSelection,selection,...rest})=>rest;
  if(JSON.stringify(fixed(previousPlan))!==JSON.stringify(fixed(plan)))
    throw new Error('Fixed experiment settings changed. Choose a new output directory.');
} catch(error) {if(error.code!=='ENOENT')throw error;}
await writeFile(join(directory,'plan.json'),JSON.stringify(plan,null,2)+'\n');
await writeFile(join(directory,'driver.pid'),String(process.pid)+'\n');
await writeFile(join(directory,'source.json'),JSON.stringify({completed:false})+'\n');
let runs = [];
try {runs=JSON.parse(await readFile(join(directory,'runs.json'),'utf8'));}
catch(error) {if(error.code!=='ENOENT')throw error;}
console.log(`Resuming ${runs.filter(run=>run.exitCode===0).length} completed CUDA trials.`);
let child, stopped = false;
const stop = () => {stopped=true; child?.kill('SIGINT');};
process.on('SIGINT',stop); process.on('SIGTERM',stop);
process.stdin.on('data',chunk=>{if(String(chunk).includes('stop'))stop();});
process.stdin.unref?.();
const pause = ms => new Promise(resolve=>setTimeout(resolve,ms));
const key = ({width,seed,learningRate,weightDecay}) => `w${width}-s${seed}-lr${learningRate}-wd${weightDecay}`;

function readHistory(text) {
  const [header,...lines] = text.trim().split('\n'), fields=header.split(',');
  return lines.map(line=>Object.fromEntries(line.split(',').map((value,i)=>[fields[i],Number(value)])));
}
function metrics(rows) {
  const first = (name,value) => rows.find(row=>row[name]>=value)?.completed_epochs ?? null;
  const final=rows.at(-1);
  return {firstTrain99:first('train_accuracy',.99),firstTest90:first('validation_accuracy',.9),firstTest99:first('validation_accuracy',.99),
    peakTrain:Math.max(...rows.map(row=>row.train_accuracy)),peakTest:Math.max(...rows.map(row=>row.validation_accuracy)),
    finalTrain:final.train_accuracy,finalTest:final.validation_accuracy,finalTestLoss:final.validation_cross_entropy,completedEpochs:final.completed_epochs};
}
// A verified binary serves the widths it lists, by default only W1024.
async function trainer(width) {
  while(!stopped) {
    try {const ready=JSON.parse(await readFile(readyPath,'utf8')); if(ready.verified&&(ready.widths??[1024]).includes(width)) {await stat(ready.binary);return ready.binary;}}
    catch(error) {if(error.code!=='ENOENT')throw error;}
    if(width!==1024)return original;
    await pause(1000);
  }
  throw new Error('Search stopped.');
}
async function run(config,epochs,stage) {
  if(stopped)throw new Error('Search stopped.');
  const reusable=runs.find(run=>key(run)===key(config)&&run.exitCode===0&&Number.isFinite(run.completedEpochs)&&(Number.isFinite(run.firstTest99)||run.completedEpochs>=epochs));
  if(reusable)return reusable;
  const completed=runs.find(run=>key(run)===key(config)&&run.stage===stage&&run.exitCode===0&&Number.isFinite(run.completedEpochs)&&(run.budget>=epochs||run.firstTest99!==null));
  if(completed)return completed;
  const name=`${stage}-${key(config)}`, location=join(directory,name);
  await mkdir(location,{recursive:true});
  const binary=await trainer(config.width);
  const args=['grok-simple','--model','tied-rotary-residual-dense-mlp','--width',String(config.width),'--depth','1','--activation','relu','--encoding','one-hot',
    '--dense-embedding-initialization','normal-width','--operation','div','--modulus','97','--train-fraction','0.5','--learning-rate',String(config.learningRate),
    '--weight-decay',String(config.weightDecay),'--seed',String(config.seed),'--batch','512','--epochs',String(epochs),'--report-every','10','--stop-validation-accuracy','0.99','--output',location];
  await writeFile(join(location,'arguments.json'),JSON.stringify({binary,args,config,stage},null,2)+'\n');
  const log=await open(join(location,'stdout.log'),'w');
  const start=Date.now();
  child=spawn(binary,args,{cwd:root,stdio:['ignore',log.fd,log.fd]});
  const code=await new Promise((resolve,reject)=>{child.once('error',reject);child.once('exit',resolve);});
  await log.close(); child=undefined;
  const record={...config,stage,budget:epochs,directory:location,binary,exitCode:code,wallSeconds:(Date.now()-start)/1000};
  try {Object.assign(record,metrics(readHistory(await readFile(join(location,'loss.csv'),'utf8'))));}catch(error){record.error=String(error);}
  runs=runs.filter(run=>key(run)!==key(config)||run.stage!==stage);
  runs.push(record);
  const temporary=join(directory,'runs.next.json');
  await writeFile(temporary,JSON.stringify(runs,null,2)+'\n');
  await rename(temporary,join(directory,'runs.json'));
  console.log(`${runs.length}: ${name} -> test99=${record.firstTest99??'not reached'}, peak=${(100*(record.peakTest??0)).toFixed(2)}%, exit=${code}`);
  if(stopped)throw new Error('Search stopped.');
  if(code!==0||record.error)throw new Error(`CUDA trial failed: ${name}. See ${location}/stdout.log.`);
  return record;
}
const baseline = width=>({width,seed:plan.screenSeed,...plan.baseline});
for(const width of plan.widths) {
  const candidates=plan.learningRates.flatMap(learningRate=>plan.weightDecays.map(weightDecay=>({width,seed:plan.screenSeed,learningRate,weightDecay})));
  candidates.sort((a,b)=>Number(key(b)===key(baseline(width)))-Number(key(a)===key(baseline(width))));
  for(const config of candidates) await run(config,screenBudget(width,runs,plan),'screen');
}
await writeFile(join(directory,'screen-complete.json'),JSON.stringify({completed:true,runs:runs.length})+'\n');
const results=[];
for(const width of plan.widths) {
  const screened=runs.filter(run=>eligible(run,plan)&&run.width===width&&run.stage==='screen'&&run.exitCode===0).sort(rank);
  const selected=plan.worstSeedSelection?screened:[...screened.slice(0,plan.finalistsPerWidth),baseline(width)];
  const unique=[...new Map(selected.map(run=>[`${run.learningRate}/${run.weightDecay}`,run])).values()];
  const profiles=[];
  for(const candidate of unique) {
    const confirmed=[];
    const budget=plan.worstSeedSelection?Math.min(plan.confirmationEpochs,robustBound(width,runs,plan)):plan.confirmationEpochs;
    for(const seed of plan.confirmationSeeds) {
      const config={width,seed,learningRate:candidate.learningRate,weightDecay:candidate.weightDecay};
      const previous=runs.find(run=>key(run)===key(config)&&run.exitCode===0&&run.firstTest99!==null);
      confirmed.push(previous??await run(config,budget,'confirm'));
    }
    const successes=confirmed.filter(run=>run.exitCode===0&&run.firstTest99!==null);
    const epochs=confirmed.map(run=>run.firstTest99??plan.confirmationEpochs+1).sort((a,b)=>a-b);
    profiles.push({learningRate:candidate.learningRate,weightDecay:candidate.weightDecay,successes:successes.length,maximumEpoch:Math.max(...epochs),maximumWallSeconds:Math.max(...confirmed.map(run=>run.wallSeconds)),medianEpoch:epochs[Math.floor(epochs.length/2)],runs:confirmed});
  }
  profiles.sort((a,b)=>b.successes-a.successes||(plan.worstSeedSelection?a.maximumEpoch-b.maximumEpoch:0)||a.medianEpoch-b.medianEpoch);
  results.push({width,profiles,winner:profiles[0].successes===plan.confirmationSeeds.length?profiles[0]:null});
  await writeFile(join(directory,'summary.json'),JSON.stringify({plan,runs: runs.length,results},null,2)+'\n');
}
const hash=async path=>createHash('sha256').update(await readFile(path)).digest('hex');
const ready=JSON.parse(await readFile(readyPath,'utf8'));
const binaries={};
for(const binary of new Set(runs.map(run=>run.binary)))binaries[binary]=await hash(binary);
await writeFile(join(directory,'source.json'),JSON.stringify({originalBinarySha256:await hash(original),wideBinarySha256:await hash(ready.binary),binarySha256:binaries,sourceSha256:await hash(new URL(import.meta.url)),completed:true},null,2)+'\n');
console.log('COMPLETE: optimizer grid and independent-seed confirmations.');
