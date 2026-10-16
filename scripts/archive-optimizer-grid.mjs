// Preserve metrics and compressed native histories without large checkpoints.
// Also archives late-generalization searches, whose summaries list no profiles.
import {mkdir, readFile, writeFile, copyFile} from 'node:fs/promises';
import {resolve, join, basename} from 'node:path';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {ShortGrokking} from '../web/own-channel/dist/model.js';

const source=resolve(process.argv[2]??'target/optimizer-grid/search');
const destination=resolve(process.argv[3]??'docs/grokking-results/dense-optimizer-grid');
const provenance=JSON.parse(await readFile(join(source,'source.json'),'utf8'));
if(!provenance.completed)throw new Error('Finish the CUDA grid before archiving.');
const summary=JSON.parse(await readFile(join(source,'summary.json'),'utf8'));
const runs=JSON.parse(await readFile(join(source,'runs.json'),'utf8'));
if(runs.some(run=>run.exitCode!==0||run.error))throw new Error('A CUDA trial failed.');
await mkdir(destination,{recursive:true});
const manifest=[];
const classified=new Map();
const archivedDirectories=new Map();
const identity=run=>`${run.stage}/${run.width}/${run.seed}/${run.learningRate}/${run.weightDecay}`;
for(const run of runs) {
  const name=basename(run.directory),location=join(destination,name);
  await mkdir(location,{recursive:true});
  const history=await readFile(join(run.directory,'loss.csv'));
  const [header,...lines]=history.toString().trim().split('\n'),fields=header.split(',');
  const criterion=new ShortGrokking();
  for(const line of lines) {
    const row=Object.fromEntries(line.split(',').map((value,index)=>[fields[index],Number(value)]));
    criterion.observe({epoch:row.completed_epochs,trainAccuracy:row.train_accuracy,heldoutAccuracy:row.validation_accuracy});
  }
  run.grokking={plateau:criterion.plateau,perfect:criterion.perfect,riseEpochs:criterion.rise,event:criterion.event};
  classified.set(identity(run),run.grokking);
  archivedDirectories.set(identity(run),name);
  await writeFile(join(location,'loss.csv.gz'),gzipSync(history,{level:9}));
  await copyFile(join(run.directory,'arguments.json'),join(location,'arguments.json'));
  await copyFile(join(run.directory,'run.txt'),join(location,'run.txt'));
  manifest.push({directory:name,historySha256:createHash('sha256').update(history).digest('hex')});
  run.directory=name;
}
for(const result of summary.results) {
  for(const profile of [...(result.profiles??[]),...(result.winner?[result.winner]:[])])
    for(const run of profile.runs) {
      const key=identity(run),location=archivedDirectories.get(key);
      if(!location)throw new Error(`Missing archived history: ${key}`);
      run.grokking=classified.get(key);run.directory=location;
    }
}
// Paths in the compact results refer to the archived evidence.
const portable=object=>JSON.parse(JSON.stringify(object,(_key,value)=>typeof value==='string'&&value.startsWith(source+'/')?basename(value):value));
for(const [name,object] of [['summary.json',summary],['runs.json',runs],['source.json',provenance],['histories.json',manifest]])
  await writeFile(join(destination,name),JSON.stringify(portable(object),null,2)+'\n');
const fields=['stage','width','seed','learningRate','weightDecay','budget','completedEpochs','firstTrain99','firstTest90','firstTest99','peakTest','finalTest','wallSeconds'];
await writeFile(join(destination,'runs.csv'),[fields.join(','),...runs.map(run=>fields.map(field=>run[field]??'').join(','))].join('\n')+'\n');
console.log(`Archived ${runs.length} CUDA trials to ${destination}.`);
