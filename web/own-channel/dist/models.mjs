import {P, Random, initialWeights as originalWeights, predict as originalPredict} from './model.mjs';

export const MODELS = {
  original: {id: 'original', title: 'Own-channel', subtitle: 'depth 5', width: 64, activation: 'relu', seed: 7, reportEvery: 250, parameters: 18880, input: 12416, readout: 12416, gain: 18624, gains: 4},
  concat: {id: 'concat', title: 'Shared E + concat', subtitle: 'depth 5', width: 64, activation: 'relu', seed: 7, reportEvery: 250, parameters: 14656, embedding: 6208, projection: 6208, gain: 14400, gains: 4, readout: 0, tiedEmbedding: true},
  residual: {id: 'residual', title: 'Plateau control', subtitle: 'dense residual', width: 64, activation: 'relu', seed: 7, reportEvery: 250, parameters: 84160, input: 12416, readout: 77952, gain: 12416, blocks: 2, expansion: 4},
  neighbor: {id: 'neighbor', title: 'Neighbor mesh', subtitle: 'Square · 2 × 3', width: 128, activation: 'square', seed: 11, reportEvery: 10, parameters: 38784, input: 24832, readout: 24832, gain: 37248, blocks: 2, meshDepth: 3, parents: 2, connectivity: 'neighbor', plateauEpochs: 25, riseEpochs: 250},
  'neighbor-concat': {id: 'neighbor-concat', title: 'Neighbor mesh', subtitle: 'Shared E · Square · 2 × 3', width: 128, activation: 'square', seed: 11, reportEvery: 10, parameters: 46720, embedding: 12416, projection: 12416, readout: 0, gain: 45184, blocks: 2, meshDepth: 3, parents: 2, connectivity: 'neighbor', plateauEpochs: 25, riseEpochs: 250, tiedEmbedding: true},
  'neighbor-rotary': {id: 'neighbor-rotary', title: 'Rotate90', subtitle: 'Square · 2 × 3', width: 128, activation: 'square', seed: 11, reportEvery: 10, parameters: 13952, embedding: 12416, readout: 0, gain: 12416, blocks: 2, meshDepth: 3, parents: 2, connectivity: 'neighbor', plateauEpochs: 25, riseEpochs: 250, tiedEmbedding: true, rotaryEmbedding: true},
  'neighbor-rotary-wide': {id: 'neighbor-rotary-wide', title: 'Rotate90', subtitle: 'Square · 2 × 3', width: 1024, activation: 'square', seed: 11, reportEvery: 10, parameters: 111616, embedding: 99328, readout: 0, gain: 99328, blocks: 2, meshDepth: 3, parents: 2, connectivity: 'neighbor', plateauEpochs: 25, riseEpochs: 250, tiedEmbedding: true, rotaryEmbedding: true},
};

export function modelInfo(id) {
  if (!Object.hasOwn(MODELS, id)) throw new Error('Unknown model.');
  return MODELS[id];
}

export function initialModelWeights(seed = 7, id = 'concat') {
  const model = modelInfo(id), W = model.width;
  if (id === 'original') return originalWeights(seed);
  const random = new Random(BigInt(seed) ^ 0x73696d706c650004n), weights = new Float32Array(model.parameters);
  const groups = model.rotaryEmbedding ? [[0, P * W, P]] : model.tiedEmbedding ? [[0, P * W, P], [model.projection, 2 * W * W, 2 * W]] : id === 'neighbor' ? [[0, 2 * P * W, 2 * P], [model.readout, W * P, W]] : [
    [0, 2 * P * W, 2 * P], [12416, 16384, W], [28800, 16384, 4 * W],
    [45184, 16384, W], [61568, 16384, 4 * W], [77952, W * P, W],
  ];
  for (const [start, count, fan] of groups) {
    const bound = Math.fround(1 / Math.fround(Math.sqrt(fan)));
    for (let i = start; i < start + count; i++) weights[i] = Math.fround((2 * random.unit() - 1) * bound);
  }
  if (id === 'concat') weights.fill(1, model.gain);
  if (model.meshDepth) for (let i = model.gain; i < weights.length; i += model.parents) weights[i] = 1;
  return weights;
}

export function inputMatrix(weights, id = 'concat') {
  const model = modelInfo(id), W = model.width;
  if (!model.tiedEmbedding) return weights.subarray(0, 2 * P * W);
  const result = new Float64Array(2 * P * W);
  if (model.rotaryEmbedding) {
    result.set(weights.subarray(0, P * W));
    for (let k = 0; k < P; k++) for (let c = 0; c < W; c++) result[(P + k) * W + c] = (c % 2 ? 1 : -1) * weights[k * W + (c ^ 1)];
    return result;
  }
  for (let operand = 0; operand < 2; operand++) for (let k = 0; k < P; k++) for (let j = 0; j < W; j++) {
    const embedding = weights[k * W + j];
    for (let c = 0; c < W; c++) result[(operand * P + k) * W + c] += embedding * weights[model.projection + (operand * W + j) * W + c];
  }
  return result;
}

export function readoutIndex(id, channel, answer) {
  const model = modelInfo(id), W = model.width;
  return model.tiedEmbedding ? answer * W + channel : model.readout + channel * P + answer;
}

export function rms(values) {
  const inverse = 1 / Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length + 1e-5);
  return {inverse, values: Float64Array.from(values, value => value * inverse)};
}

export function meshStride(model, layer) {
  return model.parents === 1 ? 0 : model.connectivity === 'neighbor' ? 1 : 2 ** (layer % Math.floor(Math.log2(model.width)));
}

export function predictModel(weights, a, b, id = 'concat', projected) {
  if (id === 'original') return originalPredict(weights, a, b);
  const model = modelInfo(id), W = model.width, input = projected ?? inputMatrix(weights, id);
  let hidden = Float64Array.from({length: W}, (_, c) => input[a * W + c] + input[(P + b) * W + c]);
  const rawInput = hidden.slice(), stages = [];
  if (id === 'concat') {
    hidden = hidden.map(value => Math.max(0, value));
    const inputHidden = hidden.slice();
    for (let layer = 0; layer < model.gains; layer++) hidden = hidden.map((value, c) => Math.max(0, value * weights[model.gain + layer * W + c]));
    return output(weights, hidden, id, {inputHidden, rawInput, stages});
  }
  for (let block = 0; block < model.blocks; block++) {
    const before = hidden.slice(), normalized = rms(hidden);
    if (model.expansion) {
      const expanded = new Float64Array(W * model.expansion), activated = new Float64Array(expanded.length), branch = new Float64Array(W), at = model.gain + block * 2 * W * expanded.length;
      for (let c = 0; c < W; c++) for (let j = 0; j < expanded.length; j++) expanded[j] += normalized.values[c] * weights[at + c * expanded.length + j];
      expanded.forEach((value, j) => { activated[j] = Math.max(0, value); });
      for (let j = 0; j < expanded.length; j++) for (let c = 0; c < W; c++) branch[c] += activated[j] * weights[at + W * expanded.length + j * W + c];
      hidden = hidden.map((value, c) => value + branch[c]);
      stages.push({before, normalized, expanded, activated, after: hidden.slice()});
      continue;
    }
    let branch = normalized.values;
    const values = [branch.slice()];
    for (let layer = 0; layer < model.meshDepth; layer++) {
      const index = block * model.meshDepth + layer, stride = meshStride(model, index);
      if (layer) branch = branch.map(value => model.activation === 'square' ? value * value : Math.max(0, value));
      branch = Float64Array.from({length: W}, (_, c) => {
        const at = model.gain + (index * W + c) * model.parents;
        return branch[c] * weights[at] + (model.parents === 2 ? branch[(c + stride) % W] * weights[at + 1] : 0);
      });
      values.push(branch.slice());
    }
    hidden = Float64Array.from(hidden, (value, c) => value + branch[c]);
    stages.push({before, normalized, values, after: hidden.slice()});
  }
  const final = rms(hidden);
  return output(weights, final.values, id, {inputHidden: rawInput, rawInput, stages, finalBefore: hidden, final});
}

function output(weights, hidden, id, details) {
  const W = modelInfo(id).width;
  const logits = new Float64Array(P);
  for (let c = 0; c < W; c++) for (let k = 0; k < P; k++) logits[k] += hidden[c] * weights[readoutIndex(id, c, k)];
  return {answer: logits.indexOf(Math.max(...logits)), hidden, logits, ...details};
}
