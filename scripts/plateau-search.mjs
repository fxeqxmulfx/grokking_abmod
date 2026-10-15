// Searches widths for the longest memorization plateau: training accuracy of
// at least 99% while held-out accuracy stays at most 10%, before the first 90%
// held-out crossing (ShortGrokking, the website's rule). Trials use the
// optimizer grid's native CUDA settings and 10000-epoch budget, and stop at
// 99% held-out accuracy. Seed 11 screens every configuration. Per width, the
// K screens with the longest plateaus of at least 25 epochs are confirmed on
// seeds 17 and 23, and so are the K longest among screens that generalize and
// among delayed ones (short rise after the plateau). Conclusive trials of the
// optimizer grid and the late-generalization searches are reused. CUDA trials
// are deterministic, so several run at once.
// Usage: node scripts/plateau-search.mjs <output> <widths> <learning rates> <weight decays> [--seeds S] [--jobs N] [--confirm K] [--dry-run]
// Widths, rates, decays and screening seeds (default 11) are comma-separated
// lists. Invocations with other lists extend the same output; its summary
// covers every run there.
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {existsSync, mkdirSync, readFileSync, writeFileSync, renameSync, rmSync, openSync, closeSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {ShortGrokking} from '../web/own-channel/dist/model.js';

const root=resolve(new URL('..',import.meta.url).pathname);
const [output,widthsText,ratesText,decaysText,...flags]=process.argv.slice(2);
if(!decaysText)throw new Error('Usage: node scripts/plateau-search.mjs <output> <widths> <learning rates> <weight decays> [--seeds S] [--jobs N] [--confirm K] [--dry-run]');
const option=(name,fallback)=>flags.includes(name)?Number(flags[flags.indexOf(name)+1]):fallback;
const list=text=>text.split(',').map(Number);
const directory=resolve(output),epochs=10000,seeds=[11,17,23],minimum=25;
const widths=list(widthsText),learningRates=list(ratesText),weightDecays=list(decaysText);
const jobs=option('--jobs',1),confirmations=option('--confirm',2),dryRun=flags.includes('--dry-run');
const screenSeeds=flags.includes('--seeds')?list(flags[flags.indexOf('--seeds')+1]):[seeds[0]];
// verify-model checks every product and the whole model at shape-generic sizes, so widths beyond the grid's run too.
const ready=JSON.parse(readFileSync(join(root,'target/optimizer-grid/cuda-ready.json'),'utf8'));
if(!ready.verified)throw new Error('No verified CUDA binary.');
const previous=['search','w16-latest','w128-latest'].map(name=>join(root,'target/optimizer-grid',name,'runs.json'))
  .flatMap(path=>existsSync(path)?JSON.parse(readFileSync(path,'utf8')):[]);
const key=({width,seed,learningRate,weightDecay})=>`w${width}-s${seed}-lr${learningRate}-wd${weightDecay}`;
const conclusive=run=>run?.exitCode===0&&(run.firstTest99!==null||run.completedEpochs>=epochs);
function history(location) {
  const [header,...lines]=readFileSync(join(location,'loss.csv'),'utf8').trim().split('\n'),fields=header.split(',');
  return lines.map(line=>Object.fromEntries(line.split(',').map((value,i)=>[fields[i],Number(value)])));
}
function metrics(rows) {
  const first=(name,value)=>rows.find(row=>row[name]>=value)?.completed_epochs??null;
  return {firstTrain99:first('train_accuracy',.99),firstTest90:first('validation_accuracy',.9),firstTest99:first('validation_accuracy',.99),
    peakTrain:Math.max(...rows.map(row=>row.train_accuracy)),peakTest:Math.max(...rows.map(row=>row.validation_accuracy)),completedEpochs:rows.at(-1).completed_epochs};
}
function grokking(rows) {
  const criterion=new ShortGrokking();
  for(const row of rows)criterion.observe({epoch:row.completed_epochs,trainAccuracy:row.train_accuracy,heldoutAccuracy:row.validation_accuracy});
  return {plateau:criterion.plateau,perfect:criterion.perfect,riseEpochs:criterion.rise,event:criterion.event};
}

mkdirSync(directory,{recursive:true});
let runs=[];
try {runs=JSON.parse(readFileSync(join(directory,'runs.json'),'utf8'));}
catch(error) {if(error.code!=='ENOENT')throw error;}
const save=(name,object)=>{writeFileSync(join(directory,name+'.next'),JSON.stringify(object,null,2)+'\n');renameSync(join(directory,name+'.next'),join(directory,name));};
writeFileSync(join(directory,'source.json'),JSON.stringify({completed:false})+'\n');
// Keeps the configurations that still need a CUDA trial, after reusing earlier conclusive ones.
function pending(configs) {
  return configs.filter(config=>{
    if(runs.some(run=>key(run)===key(config)&&conclusive(run)))return false;
    const reused=previous.filter(run=>key(run)===key(config)&&conclusive(run)).sort((a,b)=>b.completedEpochs-a.completedEpochs)[0];
    if(!reused)return true;
    runs.push({...reused,grokking:grokking(history(reused.directory))});
    save('runs.json',runs);
    return false;
  });
}
function trial(config) {
  const location=join(directory,`${config.stage}-${key(config)}`);
  rmSync(location,{recursive:true,force:true});
  mkdirSync(location,{recursive:true});
  const args=['grok-simple','--model','tied-rotary-residual-dense-mlp','--width',String(config.width),'--depth','1','--activation','relu','--encoding','one-hot',
    '--dense-embedding-initialization','normal-width','--operation','div','--modulus','97','--train-fraction','0.5','--learning-rate',String(config.learningRate),
    '--weight-decay',String(config.weightDecay),'--seed',String(config.seed),'--batch','512','--epochs',String(epochs),'--report-every','10','--stop-validation-accuracy','0.99','--output',location];
  writeFileSync(join(location,'arguments.json'),JSON.stringify({binary:ready.binary,args,config,stage:config.stage},null,2)+'\n');
  const log=openSync(join(location,'stdout.log'),'w'),start=Date.now();
  const child=spawn(ready.binary,args,{cwd:root,stdio:['ignore',log,log]});
  return new Promise((done,fail)=>{
    child.on('error',fail);
    child.on('close',status=>{
      closeSync(log);
      if(status!==0)return fail(new Error(`CUDA trial failed: ${location}/stdout.log`));
      const rows=history(location);
      done({...config,budget:epochs,directory:location,binary:ready.binary,exitCode:status,wallSeconds:(Date.now()-start)/1000,...metrics(rows),grokking:grokking(rows)});
    });
  });
}
let started=0;
async function execute(configs) {
  const queue=pending(configs);
  if(dryRun) {started+=queue.length;return;}
  await Promise.all(Array.from({length:Math.min(jobs,queue.length)},async()=>{
    for(let config;(config=queue.shift());) {
      const record=await trial(config),{plateau,event}=record.grokking;
      runs=runs.filter(run=>key(run)!==key(config)).concat(record);
      save('runs.json',runs);
      console.log(`${++started}: ${key(config)}: train99 ${record.firstTrain99??'-'}, plateau ${plateau.epochs}${plateau.start===null?'':` (${plateau.start}-${plateau.end})`}, rise ${record.grokking.riseEpochs??'-'}, test99 ${record.firstTest99??'-'}${event?.delayed?' delayed':''}, ${record.wallSeconds.toFixed(0)} s`);
    }
  }));
}
const plateauOf=run=>run.grokking.plateau.epochs;
await execute(widths.flatMap(width=>learningRates.flatMap(learningRate=>weightDecays.flatMap(weightDecay=>screenSeeds.map(seed=>({width,seed,learningRate,weightDecay,stage:'plateau'}))))));
// Confirms the longest screened plateaus of each searched width, also those of earlier invocations.
const chosen=width=>{
  const screens=runs.filter(run=>run.width===width&&run.seed===seeds[0]&&conclusive(run)&&plateauOf(run)>=minimum).sort((a,b)=>plateauOf(b)-plateauOf(a));
  return [...new Set([screens,screens.filter(run=>run.firstTest99!==null),screens.filter(run=>run.grokking.event?.delayed)].flatMap(group=>group.slice(0,confirmations)))];
};
await execute(widths.flatMap(width=>chosen(width).flatMap(screen=>seeds.slice(1).map(seed=>({width,seed,learningRate:screen.learningRate,weightDecay:screen.weightDecay,stage:'plateau'})))));
if(dryRun) {console.log(`${started} trials to run.`);process.exit(0);}

// Held-out accuracy when training accuracy first reaches 99%; near chance once memorization outruns generalization.
const memorized=new Map();
const testAtTrain99=run=>{
  if(!memorized.has(run.directory))memorized.set(run.directory,history(run.directory).find(row=>row.train_accuracy>=.99)?.validation_accuracy??null);
  return memorized.get(run.directory);
};
const describe=run=>({seed:run.seed,firstTrain99:run.firstTrain99,testAtTrain99:testAtTrain99(run),firstTest90:run.firstTest90??null,firstTest99:run.firstTest99,
  completedEpochs:run.completedEpochs,plateau:run.grokking.plateau,riseEpochs:run.grokking.riseEpochs,delayed:run.grokking.event?.delayed??false});
const shortest=configuration=>Math.min(...configuration.runs.map(run=>run.plateau.epochs));
const longest=configurations=>configurations.sort((a,b)=>shortest(b)-shortest(a))[0]??null;
const results=[...new Set(runs.map(run=>run.width))].sort((a,b)=>a-b).map(width=>{
  const own=runs.filter(run=>run.width===width&&conclusive(run)),groups=new Map();
  for(const run of own)groups.set(`${run.learningRate}/${run.weightDecay}`,[...(groups.get(`${run.learningRate}/${run.weightDecay}`)??[]),run]);
  const configurations=[...groups.values()].map(group=>({learningRate:group[0].learningRate,weightDecay:group[0].weightDecay,
    runs:seeds.map(seed=>group.find(run=>run.seed===seed)).filter(Boolean).map(describe)})).sort((a,b)=>a.learningRate-b.learningRate||a.weightDecay-b.weightDecay);
  const complete=configurations.filter(configuration=>configuration.runs.length===seeds.length);
  const pick=run=>run&&{learningRate:run.learningRate,weightDecay:run.weightDecay,...describe(run)};
  return {width,lowestTestAtTrain99:pick(own.filter(run=>testAtTrain99(run)!==null).toSorted((a,b)=>testAtTrain99(a)-testAtTrain99(b))[0]),
    longestRun:pick(own.toSorted((a,b)=>plateauOf(b)-plateauOf(a))[0]),
    longestGeneralizedRun:pick(own.filter(run=>run.firstTest99!==null).toSorted((a,b)=>plateauOf(b)-plateauOf(a))[0]),
    robust:longest(complete.filter(configuration=>configuration.runs.every(run=>run.firstTest99!==null))),
    robustDelayed:longest(complete.filter(configuration=>configuration.runs.every(run=>run.delayed))),configurations};
});
const onset=test=>results.find(test)?.width??null;
save('summary.json',{epochs,seeds,minimumPlateauEpochs:minimum,onset:{
  plateau:onset(result=>result.longestRun?.plateau.epochs>=minimum),
  robustGeneralizingPlateau:onset(result=>result.robust&&shortest(result.robust)>=minimum),
  robustDelayed:onset(result=>result.robustDelayed!==null)},results});
const hash=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
const binaries={};
for(const binary of new Set(runs.map(run=>run.binary)))binaries[binary]=hash(binary);
save('source.json',{binarySha256:binaries,sourceSha256:hash(new URL(import.meta.url)),completed:true});
for(const result of results) {
  const robust=result.robust,delayed=result.robustDelayed;
  const lowest=result.lowestTestAtTrain99;
  console.log(`W${result.width}: `+(lowest?`held-out ${(100*lowest.testAtTrain99).toFixed(1)}% at train 99% (lr ${lowest.learningRate}, wd ${lowest.weightDecay}, seed ${lowest.seed}), `:'')+
    `longest ${result.longestRun?.plateau.epochs??0}`+
    (robust?`, robust ${robust.runs.map(run=>run.plateau.epochs).join('/')} at lr ${robust.learningRate}, wd ${robust.weightDecay}`:'')+
    (delayed?`, delayed ${delayed.runs.map(run=>run.plateau.epochs).join('/')} at lr ${delayed.learningRate}, wd ${delayed.weightDecay}`:''));
}
