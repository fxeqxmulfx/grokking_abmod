import {mkdir, writeFile} from 'node:fs/promises';
import {resolve, join} from 'node:path';
import {MODEL_IDS} from '../dist/types.js';
import {modelInfo, initialModelWeights, predictModel, inputMatrix} from '../dist/models.js';
import {dataset, Sampler} from '../dist/model.js';
import {pytorchSource} from '../dist/pytorch-export.js';
import {modelLossAndGradient} from './models-reference.ts';
import {adamStep} from './reference.ts';

const directory = resolve(process.argv[2] ?? '/tmp/grokking-pytorch-fixtures');
await mkdir(directory, {recursive: true});
const bytes = (values: Float32Array | Float64Array | Uint32Array) => new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
for (const model of MODEL_IDS) {
  const location = join(directory, model); await mkdir(location, {recursive: true});
  const info = modelInfo(model), seed = 11, {width, parameters, learningRate, weightDecay} = info, weights = initialModelWeights(seed, model), data = dataset(seed);
  const ids = [0, 19, 4655, 4656, 9311], projected = inputMatrix(weights, model);
  const logits = Float64Array.from(ids.flatMap(id => Array.from(predictModel(weights, data[id * 3], data[id * 3 + 1], model, projected).logits)));
  const sample = new Sampler(seed).sample().slice(0, 7), oracle = modelLossAndGradient(weights, data, sample, model);
  const expected = Float64Array.from(weights), moments = new Float64Array(parameters * 2);
  for (let step = 1; step <= 3; step++) adamStep(expected, modelLossAndGradient(expected, data, sample, model).gradient, moments, step, info);
  const sampler = new Sampler(seed), batches = Array.from({length: 21}, () => Array.from(sampler.sample()));
  const sources: [string, string | Uint8Array][] = [
    ['model.py', pytorchSource({model, seed, maxEpochs: 17})], ['initial.f32', bytes(weights)], ['data.u32', bytes(data)],
    ['logits.f64', bytes(logits)], ['gradient.f64', bytes(oracle.gradient)], ['adam.f64', bytes(expected)],
    ['reference.json', JSON.stringify({width, seed, parameters, learningRate, weightDecay, ids, sample: Array.from(sample), loss: oracle.loss, batches}, null, 2) + '\n'],
  ];
  for (const [name, value] of sources) await writeFile(join(location, name), value);
}
console.log(`Exported all ${MODEL_IDS.length} sources and independent f64 fixtures to ${directory}.`);
