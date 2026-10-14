import test from 'node:test';
import assert from 'node:assert/strict';
import {eligible,screenBudget,robustBound} from './optimizer-grid-policy.mjs';
const plan={widths:[16,32,128,1024],learningRates:[.0001,.001,.01,.1,1],weightDecays:[0,.0001,.001,.01,.1,1],screenEpochs:2000,wideScreenEpochs:1000,finalistsPerWidth:2,screenRacing:true};
const run=(epoch,learningRate=.001,weightDecay=.1,width=1024)=>({width,learningRate,weightDecay,stage:'screen',exitCode:0,firstTest99:epoch});
test('only current grid values contribute to selection and racing',()=>{
  assert.equal(eligible(run(10,.003),plan),false);
  assert.equal(eligible(run(10,.001,.03),plan),false);
  assert.equal(eligible(run(10,1,.0001),plan),true);
  assert.equal(screenBudget(1024,[run(10,.003),run(20),run(270,.001,1)],plan),270);
});
test('racing retains a full budget until two successful candidates exist',()=>{
  assert.equal(screenBudget(1024,[],plan),1000);
  assert.equal(screenBudget(1024,[run(65)],plan),1000);
  assert.equal(screenBudget(1024,[run(65),{...run(10,.01),exitCode:1}],plan),1000);
  assert.equal(screenBudget(128,[run(65),run(100)],plan),2000);
});
test('the second-fastest successful crossing is a sufficient finalist bound',()=>{
  assert.equal(screenBudget(1024,[run(270),run(134,.001,1),run(40,.01,1)],plan),134);
  assert.equal(screenBudget(1024,[run(270),run(134,.001,1),run(40,.01,1),run(15,.1,1)],plan),40);
});
test('worst-seed racing uses complete groups, and excludes withdrawn values',()=>{
  const worst={...plan,worstSeedSelection:true,confirmationSeeds:[11,17,23],confirmationEpochs:10000};
  const group=(epochs,rate,decay)=>epochs.map((epoch,index)=>({...run(epoch,rate,decay),seed:worst.confirmationSeeds[index]}));
  const records=[...group([50,60,70],.003,1),...group([134,111,135],.001,1),...group([270,301,293],.001,.1)];
  assert.equal(robustBound(1024,records,worst),301);
  assert.equal(screenBudget(1024,records,worst),301);
  assert.equal(robustBound(1024,records.slice(0,5),worst),10000);
});
