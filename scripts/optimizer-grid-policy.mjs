export function eligible(run,plan) {
  return plan.widths.includes(run.width)&&plan.learningRates.includes(run.learningRate)&&plan.weightDecays.includes(run.weightDecay);
}
export function rank(a,b) {
  const successA=Number.isFinite(a.firstTest99),successB=Number.isFinite(b.firstTest99);
  if(successA!==successB)return successA?-1:1;
  return successA?a.firstTest99-b.firstTest99:b.peakTest-a.peakTest||a.finalTestLoss-b.finalTestLoss;
}
export function screenBudget(width,runs,plan) {
  const limit=width===1024?plan.wideScreenEpochs:plan.screenEpochs;
  if(!plan.screenRacing)return limit;
  if(plan.worstSeedSelection)return Math.min(limit,robustBound(width,runs,plan));
  const successful=runs.filter(run=>eligible(run,plan)&&run.width===width&&run.stage==='screen'&&run.exitCode===0&&Number.isFinite(run.firstTest99)).sort(rank);
  const bound=successful[plan.finalistsPerWidth-1]?.firstTest99;
  return bound===undefined?limit:Math.min(limit,Math.max(1,Math.ceil(bound)));
}
export function robustBound(width,runs,plan) {
  const configurations=new Map();
  for(const run of runs)if(eligible(run,plan)&&run.width===width&&run.exitCode===0&&Number.isFinite(run.firstTest99)) {
    const key=`${run.learningRate}/${run.weightDecay}`;
    const seeds=configurations.get(key)??new Map();
    seeds.set(run.seed,run.firstTest99); configurations.set(key,seeds);
  }
  const scores=[...configurations.values()].filter(seeds=>plan.confirmationSeeds.every(seed=>seeds.has(seed))).map(seeds=>Math.max(...plan.confirmationSeeds.map(seed=>seeds.get(seed)))).sort((a,b)=>a-b);
  return scores[plan.finalistsPerWidth-1]??plan.confirmationEpochs;
}
