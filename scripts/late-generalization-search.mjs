// Searches one width for configurations that reach 99% held-out accuracy on
// every seed, but as late as possible. Trials use the optimizer grid's native
// CUDA settings and 10000-epoch budget and stop at 99% held-out accuracy.
// Conclusive trials of the optimizer grids and the plateau search (reached
// 99%, or ran the full budget) are reused. A configuration is abandoned after
// its first seed that misses 99%, since it can no longer succeed on every seed.
// The summary also names the longest memorization plateau, the website's
// ShortGrokking rule, among configurations that succeed on every seed, and the
// longest among those that also grok on every seed by that rule.
// Usage: node scripts/late-generalization-search.mjs <width> <output> <learning rates> <weight decays> [--dry-run]
// Rates and decays are comma-separated lists.
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {existsSync, mkdirSync, readFileSync, writeFileSync, renameSync, rmSync, openSync, closeSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {ShortGrokking} from '../web/own-channel/dist/model.js';

const root=resolve(new URL('..',import.meta.url).pathname);
const [widthText,output,ratesText,decaysText,flag]=process.argv.slice(2);
if(!ratesText||!decaysText)throw new Error('Usage: node scripts/late-generalization-search.mjs <width> <output> <learning rates> <weight decays> [--dry-run]');
const width=Number(widthText),directory=resolve(output),epochs=10000,seeds=[11,17,23];
const list=text=>text.split(',').map(Number);
const learningRates=list(ratesText),weightDecays=list(decaysText);
const ready=JSON.parse(readFileSync(join(root,'target/optimizer-grid/cuda-ready.json'),'utf8'));
if(!ready.verified||!(ready.widths??[1024]).includes(width))throw new Error(`No verified CUDA binary serves W${width}.`);
const grid=['search','grid-64-256-512','plateau'].map(name=>join(root,'target/optimizer-grid',name,'runs.json'))
  .flatMap(path=>existsSync(path)?JSON.parse(readFileSync(path,'utf8')):[]);
const key=({width,seed,learningRate,weightDecay})=>`w${width}-s${seed}-lr${learningRate}-wd${weightDecay}`;
const conclusive=run=>run?.exitCode===0&&(run.firstTest99!==null||run.completedEpochs>=epochs);
function history(location) {
  const [header,...lines]=readFileSync(join(location,'loss.csv'),'utf8').trim().split('\n'),fields=header.split(',');
  return lines.map(line=>Object.fromEntries(line.split(',').map((value,i)=>[fields[i],Number(value)])));
}
function metrics(rows) {
  const first=(name,value)=>rows.find(row=>row[name]>=value)?.completed_epochs??null;
  return {firstTrain99:first('train_accuracy',.99),firstTest99:first('validation_accuracy',.99),
    peakTrain:Math.max(...rows.map(row=>row.train_accuracy)),peakTest:Math.max(...rows.map(row=>row.validation_accuracy)),completedEpochs:rows.at(-1).completed_epochs};
}
function grokking(location) {
  const criterion=new ShortGrokking();
  for(const row of history(location))criterion.observe({epoch:row.completed_epochs,trainAccuracy:row.train_accuracy,heldoutAccuracy:row.validation_accuracy});
  return criterion;
}

mkdirSync(directory,{recursive:true});
let runs=[];
try {runs=JSON.parse(readFileSync(join(directory,'runs.json'),'utf8'));}
catch(error) {if(error.code!=='ENOENT')throw error;}
const save=(name,object)=>{writeFileSync(join(directory,name+'.next'),JSON.stringify(object,null,2)+'\n');renameSync(join(directory,name+'.next'),join(directory,name));};
writeFileSync(join(directory,'source.json'),JSON.stringify({completed:false})+'\n');
const configurations=learningRates.flatMap(learningRate=>weightDecays.map(weightDecay=>({learningRate,weightDecay})));
let started=0;
for(const configuration of configurations) for(const seed of seeds) {
  const config={width,seed,...configuration},same=run=>run.width===width&&run.learningRate===config.learningRate&&run.weightDecay===config.weightDecay;
  if(runs.some(run=>same(run)&&seeds.indexOf(run.seed)<seeds.indexOf(seed)&&conclusive(run)&&run.firstTest99===null))break;
  if(runs.some(run=>key(run)===key(config)&&conclusive(run)))continue;
  const reused=grid.filter(run=>key(run)===key(config)&&conclusive(run)).sort((a,b)=>b.completedEpochs-a.completedEpochs)[0];
  if(reused) {runs.push(reused);save('runs.json',runs);continue;}
  if(flag==='--dry-run') {started++;continue;}
  const location=join(directory,key(config));
  rmSync(location,{recursive:true,force:true});
  mkdirSync(location,{recursive:true});
  const args=['grok-simple','--model','tied-rotary-residual-dense-mlp','--width',String(width),'--depth','1','--activation','relu','--encoding','one-hot',
    '--dense-embedding-initialization','normal-width','--operation','div','--modulus','97','--train-fraction','0.5','--learning-rate',String(config.learningRate),
    '--weight-decay',String(config.weightDecay),'--seed',String(seed),'--batch','512','--epochs',String(epochs),'--report-every','10','--stop-validation-accuracy','0.99','--output',location];
  writeFileSync(join(location,'arguments.json'),JSON.stringify({binary:ready.binary,args,config,stage:'late'},null,2)+'\n');
  const log=openSync(join(location,'stdout.log'),'w'),start=Date.now();
  const child=spawnSync(ready.binary,args,{cwd:root,stdio:['ignore',log,log]});
  closeSync(log);
  if(child.status!==0)throw new Error(`CUDA trial failed: ${location}/stdout.log`);
  const record={...config,stage:'late',budget:epochs,directory:location,binary:ready.binary,exitCode:child.status,wallSeconds:(Date.now()-start)/1000,...metrics(history(location))};
  runs=runs.filter(run=>key(run)!==key(config)).concat(record);
  save('runs.json',runs);
  console.log(`${++started}: ${key(config)}: train99 ${record.firstTrain99??'-'}, test99 ${record.firstTest99??'-'}, peak ${(100*record.peakTest).toFixed(1)}%, ${record.wallSeconds.toFixed(0)} s`);
}
if(flag==='--dry-run') {console.log(`${started} trials to run.`);process.exit(0);}
// Configurations that succeed on every seed are ranked by their earliest seed, latest first.
const results=configurations.map(({learningRate,weightDecay})=>{
  const selected=seeds.map(seed=>runs.find(run=>key(run)===key({width,seed,learningRate,weightDecay}))).filter(Boolean);
  const criteria=selected.map(run=>grokking(run.directory));
  return {learningRate,weightDecay,seeds:selected.map(run=>run.seed),firstTrain99:selected.map(run=>run.firstTrain99),firstTest99:selected.map(run=>run.firstTest99),
    plateauEpochs:criteria.map(criterion=>criterion.plateau.epochs),riseEpochs:criteria.map(criterion=>criterion.rise),
    delayed:criteria.map(criterion=>criterion.event?.delayed??false),successes:selected.filter(run=>run.firstTest99!==null).length};
});
const robust=results.filter(result=>result.successes===seeds.length).sort((a,b)=>Math.min(...b.firstTest99)-Math.min(...a.firstTest99));
// Plateaus count from 25 epochs, the website's minimum for delayed grokking, and rank by their shortest seed.
const plateaus=robust.filter(result=>Math.min(...result.plateauEpochs)>=25).sort((a,b)=>Math.min(...b.plateauEpochs)-Math.min(...a.plateauEpochs));
const grokked=plateaus.filter(result=>result.delayed.every(Boolean));
save('summary.json',{width,seeds,epochs,learningRates,weightDecays,latest:robust[0]??null,longestPlateau:plateaus[0]??null,longestDelayedPlateau:grokked[0]??null,results});
const hash=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
const binaries={};
for(const binary of new Set(runs.map(run=>run.binary)))binaries[binary]=hash(binary);
save('source.json',{binarySha256:binaries,sourceSha256:hash(new URL(import.meta.url)),completed:true});
console.log(robust.length?`Latest robust: lr ${robust[0].learningRate}, wd ${robust[0].weightDecay}, test99 ${robust[0].firstTest99.join('/')}`:'No configuration reached 99% on every seed.');
console.log(plateaus.length?`Longest robust plateau: lr ${plateaus[0].learningRate}, wd ${plateaus[0].weightDecay}, plateau ${plateaus[0].plateauEpochs.join('/')}, test99 ${plateaus[0].firstTest99.join('/')}`:'No configuration plateaus for 25 epochs on every seed.');
console.log(grokked.length?`Longest robust grokking plateau: lr ${grokked[0].learningRate}, wd ${grokked[0].weightDecay}, plateau ${grokked[0].plateauEpochs.join('/')}, test99 ${grokked[0].firstTest99.join('/')}`:'No configuration groks on every seed.');
