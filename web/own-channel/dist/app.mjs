import {P} from './model.mjs';
import {modelInfo, initialModelWeights, predictModel, inputMatrix, readoutIndex} from './models.mjs';

const $ = id => document.getElementById(id);
function experiment(model) {
  const info = modelInfo(model), seed = info.seed, initial = initialModelWeights(seed, model);
  return {model, weights: initial.slice(), initial, history: [], channel: 0, metric: info.blocks ? 'accuracy' : 'loss', group: info.meshDepth ? '0' : 'io', worker: null, running: false, ready: false, label: 'Loading', step: 0, seed, activeMs: 0, dirty: true,
    options: {model, seed, backend: 'auto', maxEpochs: 10000, captureHz: 5, stopAtTarget: true}};
}
const experiments = {'neighbor-rotary': experiment('neighbor-rotary')};
let state = experiments['neighbor-rotary'];
let W = 128;
const orange = '#dd6b3c', blue = '#49849c', ink = '#252d28', muted = '#96988e', line = '#e7e7dd';
const format = value => value === 0 ? '0' : value < .001 ? value.toExponential(2) : value.toFixed(3);
function canvas(id) {
  const element = $(id), rect = element.getBoundingClientRect(), scale = Math.min(devicePixelRatio || 1, 2);
  const width = Math.round(rect.width), height = Math.round(rect.height);
  if (element.width !== Math.round(width * scale) || element.height !== Math.round(height * scale)) { element.width = Math.round(width * scale); element.height = Math.round(height * scale); }
  const ctx = element.getContext('2d'); ctx.setTransform(scale, 0, 0, scale, 0, 0); ctx.clearRect(0, 0, width, height);
  return {ctx, width, height};
}
function text(ctx, label, x, y, align = 'left', colour = muted) { ctx.font = '10px "DM Mono", monospace'; ctx.fillStyle = colour; ctx.textAlign = align; ctx.fillText(label, x, y); }
function curve() {
  const {ctx, width, height} = canvas('curve'), pad = {l: 48, r: 14, t: 18, b: 31};
  const rows = state.history, plotW = width - pad.l - pad.r, plotH = height - pad.t - pad.b;
  const interval = Boolean(modelInfo(state.model).meshDepth) ? 50 : 250;
  const maxEpoch = Math.max(25, Math.ceil((rows[rows.length - 1]?.epoch || 0) / interval) * interval);
  const isLoss = state.metric === 'loss';
  const losses = rows.flatMap(r => [r.trainLoss, r.heldoutLoss]);
  const lo = isLoss ? Math.max(-8, Math.min(-1, Math.floor(Math.log10(Math.max(1e-8, Math.min(...losses, .1)))))) : 0;
  const hi = isLoss ? Math.max(1, Math.ceil(Math.log10(Math.max(...losses, 5)))) : 100;
  const x = value => pad.l + value / maxEpoch * plotW;
  const y = value => pad.t + plotH * (1 - ((isLoss ? Math.log10(Math.max(value, 1e-8)) : 100 * value) - lo) / (hi - lo));
  ctx.strokeStyle = line; ctx.lineWidth = 1;
  const ticks = isLoss ? Array.from({length: hi - lo + 1}, (_, i) => lo + i) : [0, 25, 50, 75, 100];
  for (const tick of ticks) {
    const py = pad.t + plotH * (1 - (tick - lo) / (hi - lo));
    ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(width - pad.r, py); ctx.stroke();
    text(ctx, isLoss ? `1e${tick}` : `${tick}%`, pad.l - 9, py + 3, 'right');
  }
  for (let i = 0; i <= 4; i++) text(ctx, Math.round(maxEpoch * i / 4).toLocaleString('en-US'), x(maxEpoch * i / 4), height - 11, 'center');
  text(ctx, 'EPOCH', width - pad.r, 10, 'right');
  ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t - 3, plotW, plotH + 6); ctx.clip();
  if (Boolean(modelInfo(state.model).meshDepth)) for (const [band, colour] of [[state.criterion?.plateau, '#ddd7c8'], [state.criterion?.perfect, '#cbbad9']]) {
    if (band?.start !== null && band?.end > band?.start) { ctx.fillStyle = colour; ctx.globalAlpha = .4; ctx.fillRect(x(band.start), pad.t, x(band.end) - x(band.start), plotH); ctx.globalAlpha = 1; }
  }
  for (const [key, colour] of isLoss ? [['trainLoss', orange], ['heldoutLoss', blue]] : [['trainAccuracy', orange], ['heldoutAccuracy', blue]]) {
    ctx.beginPath(); ctx.strokeStyle = colour; ctx.lineWidth = 2;
    rows.forEach((r, i) => i ? ctx.lineTo(x(r.epoch), y(r[key])) : ctx.moveTo(x(r.epoch), y(r[key])));
    ctx.stroke(); const last = rows[rows.length - 1];
    if (last) { ctx.beginPath(); ctx.fillStyle = colour; ctx.arc(x(last.epoch), y(last[key]), 3, 0, 2 * Math.PI); ctx.fill(); }
  }
  if (!rows.length) text(ctx, 'Start training to see the curve.', pad.l + plotW / 2, pad.t + plotH / 2, 'center');
  ctx.restore();
}
function probe() {
  const a = Number($('probe-a').value), b = Number($('probe-b').value);
  if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || a >= P || b < 1 || b >= P) { $('probe-result').textContent = 'Use integer a = 0…96 and b = 1…96.'; return null; }
  const result = predictModel(state.weights, a, b, state.model, state.projected);
  let correct = 0; while (correct * b % P !== a) correct++;
  const maximum = Math.max(...result.logits), denominator = result.logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0);
  $('prediction').textContent = result.answer;
  $('probe-result').textContent = `Correct answer: ${correct} · confidence ${(100 / denominator).toFixed(1)}% · ${correct === result.answer ? 'correct' : 'not yet correct'}`;
  return result;
}
function mesh(result, scales) {
  if (Boolean(modelInfo(state.model).meshDepth)) return neighborMesh(result, scales);
  if (state.model === 'residual') return residualMesh(result, scales);
  const GAIN = modelInfo(state.model).gain;
  const {ctx, width, height} = canvas('mesh'), left = 76, right = 10, top = 24, gap = (height - 55) / 4;
  const x = c => left + (c + .5) / W * (width - left - right), y = l => top + l * gap;
  const nodeRadius = Math.max(1.1, Math.min(3, (width - left - right) / W * .22));
  ctx.fillStyle = '#f1ede4'; ctx.fillRect(x(state.channel) - (width - left - right) / W / 2, top - 9, (width - left - right) / W, gap * 4 + 18);
  let maxGain = 1;
  for (let i = GAIN; i < state.weights.length; i++) maxGain = Math.max(maxGain, Math.abs(state.weights[i]));
  const active = Array.from(result?.inputHidden || new Float64Array(W));
  for (let l = 0; l < 5; l++) {
    if (l) for (let c = 0; c < W; c++) {
      const gain = state.weights[GAIN + (l - 1) * W + c];
      ctx.beginPath(); ctx.strokeStyle = gain >= 0 ? orange : blue;
      ctx.globalAlpha = c === state.channel ? 1 : .65;
      ctx.lineWidth = .3 + 2.5 * Math.abs(gain) / maxGain; ctx.moveTo(x(c), y(l - 1)); ctx.lineTo(x(c), y(l)); ctx.stroke();
      active[c] = Math.max(0, active[c] * gain);
    }
    ctx.globalAlpha = 1; text(ctx, l ? `Gain ${l}` : 'ReLU', 0, y(l) + 3);
    const maximum = Math.max(...active, 1e-9);
    for (let c = 0; c < W; c++) {
      ctx.beginPath(); ctx.fillStyle = colour(active[c] / maximum, 1);
      ctx.arc(x(c), y(l), nodeRadius + (c === state.channel ? .7 : 0), 0, Math.PI * 2); ctx.fill();
      if (c === state.channel) { ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.stroke(); }
    }
  }
  for (let c = 0; c < W; c += 8) text(ctx, `${c}`, x(c), height - 4, 'center');
  text(ctx, '63', x(63), height - 4, 'center');
  const gain = Array.from({length: 4}, (_, l) => Math.max(0, state.weights[GAIN + l * W + state.channel])).reduce((a, b) => a * b, 1);
  $('selected-channel').textContent = 'Channel'; $('channel-select').value = state.channel;
  $('gain-info').textContent = `Gain product: ${format(gain)}`;
  $('scale-info').textContent = `Scale: ${format(scales[state.channel])}`;
}
function rgb(value, range) {
  const t = Math.min(1, Math.abs(value) / Math.max(range, 1e-12));
  const target = value >= 0 ? [221, 107, 60] : [73, 132, 156];
  return target.map((c, i) => Math.round([249, 247, 239][i] * (1 - t) + c * t));
}
function neighborMesh(result, scales) {
  const {ctx, width, height} = canvas('mesh'), block = state.group === '1' ? 1 : 0;
  const model = modelInfo(state.model), stage = result?.stages[block], c = state.channel, counts = [4, 3, 2, 1];
  const gap = Math.max(12, (height - 82) / 3), x = layer => 35 + layer * (width - 135) / 3;
  const y = (layer, index) => 36 + (index + (4 - counts[layer]) / 2) * gap;
  const sumX = width - 23, sumY = y(3, 0), at = model.gain + block * 3 * W * 2;
  const nodes = counts.map((count, layer) => Array.from({length: count}, (_, index) => {
    const channel = (c + index) % W, raw = stage?.values[layer]?.[channel] ?? 0;
    return {x: x(layer), y: y(layer, index), channel, value: layer > 0 && layer < 3 ? raw * raw : raw};
  }));
  state.meshNodes = nodes.flat();
  let maximum = 1e-9;
  for (let layer = 1; layer <= 3; layer++) for (const node of nodes[layer]) {
    const index = at + ((layer - 1) * W + node.channel) * 2;
    maximum = Math.max(maximum, Math.abs(state.weights[index]), Math.abs(state.weights[index + 1]));
  }
  for (let layer = 1; layer <= 3; layer++) for (let index = 0; index < counts[layer]; index++) for (let parent = 0; parent < 2; parent++) {
    const from = nodes[layer - 1][index + parent], to = nodes[layer][index];
    const weight = state.weights[at + ((layer - 1) * W + to.channel) * 2 + parent];
    ctx.beginPath(); ctx.strokeStyle = weight >= 0 ? orange : blue;
    ctx.globalAlpha = .35 + .65 * Math.abs(weight) / maximum; ctx.lineWidth = .6 + 3 * Math.abs(weight) / maximum;
    ctx.moveTo(from.x + 7, from.y); ctx.lineTo(to.x - 7, to.y); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(x(0), height - 16);
  ctx.bezierCurveTo(sumX, height - 16, sumX, height - 16, sumX, sumY + 8); ctx.stroke(); ctx.setLineDash([]);
  text(ctx, `skip x[${c}]`, x(0), height - 5);
  ctx.beginPath(); ctx.moveTo(x(3) + 7, sumY); ctx.lineTo(sumX - 8, sumY); ctx.stroke();
  for (let layer = 0; layer <= 3; layer++) {
    text(ctx, ['RMSNorm', 'Square', 'Square', 'Mesh'][layer], x(layer), 13, 'center');
    const range = Math.max(...nodes[layer].map(node => Math.abs(node.value)), 1e-9);
    for (const node of nodes[layer]) {
      ctx.beginPath(); ctx.fillStyle = colour(node.value, range); ctx.arc(node.x, node.y, 6.5, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = ink; ctx.lineWidth = .7; ctx.stroke();
      text(ctx, String(node.channel), node.x - 10, node.y + 3, 'right', ink);
      if (width > 500) text(ctx, format(node.value), node.x + 10, node.y - 7);
    }
  }
  ctx.beginPath(); ctx.fillStyle = '#f8f6f1'; ctx.arc(sumX, sumY, 8, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = ink; ctx.stroke();
  text(ctx, '+', sumX, sumY + 3, 'center', ink); text(ctx, 'skip', sumX, sumY + 23, 'center');
  $('selected-channel').textContent = 'Channel'; $('channel-select').value = c;
  $('gain-info').textContent = `Block ${block + 1} · RMS ${format(stage?.normalized.inverse ?? 0)}`;
  $('scale-info').textContent = `Contribution: ${format(scales[c])}`;
  $('mesh').setAttribute('aria-label', `Neighbor residual block ${block + 1}, channel ${c}, four to three to two to one neurons, twelve learned edges and one skip connection`);
}
function residualMesh(result, scales) {
  const {ctx, width, height} = canvas('mesh'), block = state.group === '1' ? 1 : 0, stage = result?.stages[block], left = 76, right = 10;
  const x = (c, count) => left + (c + .5) / count * (width - left - right), ys = [22, height / 2, height - 25];
  const at = modelInfo(state.model).gain + block * 2 * W * 4 * W, c = state.channel;
  let maximum = 1e-9;
  for (let j = 0; j < 4 * W; j++) maximum = Math.max(maximum, Math.abs(state.weights[at + c * 4 * W + j]), Math.abs(state.weights[at + W * 4 * W + j * W + c]));
  for (let j = 0; j < 4 * W; j++) for (const side of [0, 1]) {
    const value = state.weights[side ? at + W * 4 * W + j * W + c : at + c * 4 * W + j];
    ctx.beginPath(); ctx.strokeStyle = value >= 0 ? orange : blue; ctx.globalAlpha = .15 + .55 * Math.abs(value) / maximum; ctx.lineWidth = .4 + Math.abs(value) / maximum;
    ctx.moveTo(side ? x(j, 4 * W) : x(c, W), ys[side]); ctx.lineTo(side ? x(c, W) : x(j, 4 * W), ys[side + 1]); ctx.stroke();
  }
  ctx.globalAlpha = 1; ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(x(c, W), ys[0]); ctx.bezierCurveTo(width - 2, ys[0], width - 2, ys[2], x(c, W), ys[2]); ctx.stroke(); ctx.setLineDash([]);
  for (const [layer, values, count] of [[0, stage?.normalized.values, W], [1, stage?.activated, 4 * W], [2, stage?.after, W]]) {
    text(ctx, ['RMSNorm', 'ReLU ×256', '+ skip'][layer], 0, ys[layer] + 3);
    const range = Math.max(...Array.from(values ?? [] , value => Math.abs(value)), 1e-9);
    for (let j = 0; j < count; j++) {
      ctx.beginPath(); ctx.fillStyle = colour(values?.[j] ?? 0, range); ctx.arc(x(j, count), ys[layer], count === W ? 2 : 1.2, 0, 2 * Math.PI); ctx.fill();
      if (j === c && count === W) { ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.stroke(); }
    }
  }
  $('selected-channel').textContent = 'Channel'; $('channel-select').value = c;
  $('gain-info').textContent = `Block ${block + 1} · RMS ${format(stage?.normalized.inverse ?? 0)}`;
  $('scale-info').textContent = `Contribution: ${format(scales[c])}`;
}
function colour(value, range) { return `rgb(${rgb(value, range).join(',')})`; }
const scratch = document.createElement('canvas'); scratch.width = W; scratch.height = P;
const scratchCtx = scratch.getContext('2d');
function atlasShape(type) {
  if (Boolean(modelInfo(state.model).meshDepth) && state.group !== 'io' && type !== 'r') return {rows: 3, columns: W};
  if (state.model === 'residual' && state.group !== 'io' && type !== 'r') return type === 'a' ? {rows: W, columns: 4 * W} : {rows: 4 * W, columns: W};
  return {rows: state.model === 'concat' && type === 'b' ? 2 * W : P, columns: W};
}
function atlasIndex(type, channel, row) {
  if (type === 'r') return readoutIndex(state.model, channel, row);
  if (state.model === 'concat' && type === 'b') return modelInfo(state.model).projection + row * W + channel;
  if (Boolean(modelInfo(state.model).meshDepth) && state.group !== 'io') return modelInfo(state.model).gain + ((Number(state.group) * 3 + row) * W + channel) * 2 + Number(type === 'b');
  if (modelInfo(state.model).rotaryEmbedding) return row * W + (type === 'b' ? channel ^ 1 : channel);
  if (state.model === 'residual' && state.group !== 'io') {
    const at = modelInfo(state.model).gain + Number(state.group) * 2 * W * 4 * W;
    return type === 'a' ? at + row * 4 * W + channel : at + W * 4 * W + row * W + channel;
  }
  return (row + (state.model !== 'concat' && type === 'b' ? P : 0)) * W + channel;
}
function atlasSign(type, channel) {
  return modelInfo(state.model).rotaryEmbedding && state.group === 'io' && type === 'b' && channel % 2 === 0 ? -1 : 1;
}
function atlas(id, type) {
  const {rows, columns} = atlasShape(type);
  scratch.width = columns; scratch.height = rows;
  const {ctx, width, height} = canvas(id), image = scratchCtx.createImageData(columns, rows);
  const values = new Float64Array(rows * columns); let maximum = 0;
  for (let c = 0; c < columns; c++) {
    let mean = 0;
    if (type === 'r' && $('centered').checked) for (let k = 0; k < P; k++) {
      const index = readoutIndex(state.model, c, k);
      mean += state.weights[index] - ($('weight-view').value === 'delta' ? state.initial[index] : 0);
    }
    mean /= P;
    for (let k = 0; k < rows; k++) {
      const index = atlasIndex(type, c, k);
      const value = atlasSign(type, c) * (state.weights[index] - ($('weight-view').value === 'delta' ? state.initial[index] : 0)) - mean;
      values[k * columns + c] = value; maximum = Math.max(maximum, Math.abs(value));
    }
  }
  values.forEach((value, i) => { const [r, g, b] = rgb(value, maximum); image.data.set([r, g, b, 255], i * 4); });
  scratchCtx.putImageData(image, 0, 0); ctx.imageSmoothingEnabled = false; ctx.drawImage(scratch, 0, 0, width, height);
  ctx.strokeStyle = ink; ctx.lineWidth = 1;
  if (columns !== W) ctx.strokeRect(.5, state.channel * height / rows + .5, width - 1, height / rows - 1);
  else ctx.strokeRect(state.channel * width / W, .5, Math.max(1, width / W - 1), height - 1);
  $('range-' + type).textContent = `±${format(maximum)}`;
}
function channelScales(result) {
  const GAIN = modelInfo(state.model).gain;
  const scales = new Float64Array(W);
  for (let c = 0; c < W; c++) {
    let e = 0, r = 0, mean = 0, gain = 1;
    for (let k = 0; k < 2 * P; k++) e += state.projected[k * W + c] ** 2;
    for (let k = 0; k < P; k++) mean += state.weights[readoutIndex(state.model, c, k)] / P;
    for (let k = 0; k < P; k++) r += (state.weights[readoutIndex(state.model, c, k)] - mean) ** 2;
    if (modelInfo(state.model).blocks) scales[c] = Math.abs(result?.hidden[c] ?? 0) * Math.sqrt(r);
    else { for (let l = 0; l < 4; l++) gain *= Math.max(0, state.weights[GAIN + l * W + c]); scales[c] = Math.sqrt(e * r) * gain; }
  }
  return scales;
}
function scalePlot(values) {
  const {ctx, width, height} = canvas('scales'), max = Math.max(...values, 1e-12), count = Math.min(W, Math.floor(width)), bar = width / count;
  state.scaleBins = Array.from({length: count}, (_, i) => {
    const start = Math.floor(i * W / count), end = Math.floor((i + 1) * W / count);
    let channel = start;
    for (let c = start + 1; c < end; c++) if (values[c] > values[channel]) channel = c;
    return {channel, start, end};
  });
  state.scaleBins.forEach(({channel, start, end}, i) => {
    const selected = state.channel >= start && state.channel < end, value = values[channel];
    ctx.fillStyle = selected ? ink : orange; ctx.globalAlpha = selected ? 1 : .7;
    ctx.fillRect(i * bar, (height - 20) * (1 - value / max), Math.max(1, bar - 1), (height - 20) * value / max);
  });
  ctx.globalAlpha = 1; for (let c = 0; c < W; c += W / 8) text(ctx, `${c}`, (c + .5) * width / W, height - 3, 'center');
  const total = values.reduce((a, b) => a + b, 0), top = Array.from(values).sort((a, b) => b - a).slice(0, 16).reduce((a, b) => a + b, 0);
  $('concentration').textContent = `Top 16: ${(100 * top / Math.max(total, 1e-12)).toFixed(1)}%`;
}
function draw() {
  if (state.dirty) {
    state.dirty = false;
    if (state.projectedWeights !== state.weights) { state.projected = inputMatrix(state.weights, state.model); state.projectedWeights = state.weights; }
    const visible = id => { const rect = $(id).getBoundingClientRect(); return rect.width > 0 && rect.height > 0; };
    if (visible('curve')) curve();
    const result = probe();
    if (visible('mesh') || visible('scales')) {
      const scales = channelScales(result);
      if (visible('mesh')) mesh(result, scales);
      if (visible('scales')) scalePlot(scales);
    }
    for (const [id, type] of [['input-a', 'a'], ['input-b', 'b'], ['readout', 'r']]) if (visible(id)) atlas(id, type);
  }
  requestAnimationFrame(draw);
}
function status(label, target = state) {
  target.label = label; target.running = label === 'Training' || label === 'Pausing';
  if (target !== state) return;
  $('status').textContent = label;
  $('train').innerHTML = state.running ? 'Pause training <span>Ⅱ</span>' : 'Start training <span>↗</span>';
  $('train').classList.toggle('running', state.running);
  $('train').disabled = !state.ready || label === 'Pausing';
  $('step').disabled = !state.ready || state.running;
  for (const id of ['seed', 'limit', 'stop-target', 'backend']) $(id).disabled = state.running;
}
function refresh() {
  const options = state.options, row = state.history.at(-1), model = modelInfo(state.model);
  W = model.width;
  $('seed').value = options.seed; $('limit').value = options.maxEpochs; $('backend').value = options.backend;
  $('capture').value = options.captureHz; $('stop-target').checked = options.stopAtTarget;
  $('weight-group').value = state.group; $('weight-group').hidden = !model.blocks;
  $('model-name').textContent = model.title + ' · ' + model.subtitle;
  $('model-name').title = $('model-name').textContent;
  $('model-detail').textContent = W + ' channels · ' + model.parameters.toLocaleString('en-US') + ' parameters · ' + (model.activation === 'square' ? 'Square' : 'ReLU') + ' · FP32';
  $('cadence').textContent = model.reportEvery === 10 ? 'Evaluated every epoch' : 'Evaluated every 25 epochs';
  $('step').textContent = model.reportEvery === 10 ? '+1 epoch' : '+25 epochs';
  $('atlas-r-size').textContent = W + ' × 97';
  $('channel-select').max = W - 1; $('channel-select').value = state.channel;
  const epoch = Math.floor(state.step / 10) + (state.step % 10) * 512 / 4656;
  $('epoch').textContent = epoch.toLocaleString('en-US', {maximumFractionDigits: 1});
  $('progress').value = epoch; $('progress').max = options.maxEpochs;
  $('budget-label').textContent = '/ ' + options.maxEpochs.toLocaleString('en-US');
  $('observations').textContent = state.history.length + ' observations';
  $('phase').textContent = row?.phase ?? 'Waiting for the first observation';
  for (const [id, key] of [['train-accuracy', 'trainAccuracy'], ['heldout-accuracy', 'heldoutAccuracy']]) $(id).textContent = row ? (100 * row[key]).toFixed(2) + '%' : '—';
  for (const [id, key] of [['train-loss', 'trainLoss'], ['heldout-loss', 'heldoutLoss']]) $(id).textContent = row ? format(row[key]) : '—';
  $('throughput').textContent = state.activeMs ? Math.round(state.step * 1000 / state.activeMs).toLocaleString('en-US') + ' updates / sec' : state.ready ? 'Ready when you are' : 'Preparing training engine';
  const seconds = Math.floor(state.activeMs / 1000); $('elapsed').textContent = Math.floor(seconds / 60) + ':' + String(seconds % 60).padStart(2, '0');
  const info = state.info;
  $('device').textContent = info ? info.description + (info.fallback ? ' · software adapter' : '') + (info.backend === 'webgpu' ? ' · WebGPU' : '') : state.error ? 'Training engine could not start' : 'Preparing training engine…';
  $('device').title = $('device').textContent;
  $('backend-note').textContent = info?.note ?? ''; $('backend-note').hidden = !info?.note;
  $('error').hidden = !state.error; $('error').textContent = state.error ?? ''; $('error').title = state.error ?? '';
  for (const [id, metric] of [['loss-tab', 'loss'], ['accuracy-tab', 'accuracy']]) { $(id).classList.toggle('active', state.metric === metric); $(id).setAttribute('aria-pressed', String(state.metric === metric)); }
  $('axis-note').textContent = state.metric === 'loss' ? 'Cross-entropy · logarithmic scale' : 'Complete-split accuracy';
  for (const button of document.querySelectorAll('[data-model-tab]')) {
    const active = button.dataset.modelTab === state.model; button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
  }
  const concat = state.model === 'concat', neighbor = Boolean(model.meshDepth), rotary = Boolean(model.rotaryEmbedding), block = state.group !== 'io';
  $('atlas-a-title').textContent = block ? neighbor ? 'Self edges' : 'Expansion A' : rotary || concat ? 'Shared embedding E' : 'Operand a Eₐ';
  $('atlas-b-title').textContent = block ? neighbor ? 'Neighbor edges' : 'Contraction B' : rotary ? 'Rotate90(E) · b role' : concat ? 'Concat projection A' : 'Operand b Eᵦ';
  $('atlas-r-title').textContent = model.tiedEmbedding ? 'Tied readout Eᵀ' : 'Readout R';
  for (const type of ['a', 'b']) { const shape = atlasShape(type); $('atlas-' + type + '-size').textContent = shape.rows + ' × ' + shape.columns; }
  $('atlas-a-axis').textContent = concat ? 'Residue ↓ · feature →' : block ? neighbor ? 'Layer ↓ · output channel →' : 'Channel ↓ · feature →' : 'Residue ↓ · channel →';
  $('atlas-b-axis').textContent = concat ? 'a / b features ↓ · channel →' : block ? neighbor ? 'Layer ↓ · output channel →' : 'Feature ↓ · channel →' : 'Residue ↓ · channel →';
  $('mesh-title').textContent = concat ? '64 channel paths' : 'Residual block ' + (state.group === '1' ? '2' : '1');
  $('mesh-helper').textContent = concat ? 'Edges: gain · nodes: probe activation · click to inspect' : neighbor ? '4 → 3 → 2 → 1 · edge weights · probe activations · click a node' : 'Selected channel edges · probe activations · skip connection';
  $('scales-title').textContent = concat ? 'Channel scales' : 'Probe contributions';
  $('scales').setAttribute('aria-label', `${concat ? 'Centered channel scales' : 'Probe readout contributions'} for all ${W} channels`);
  $('mesh').setAttribute('aria-label', concat ? '64 independent paths through four own-channel gain layers' : 'Selected Neighbor channel ancestors and skip connection');
  for (const type of ['a', 'b']) $('input-' + type).setAttribute('aria-label', $('atlas-' + type + '-title').textContent + ' weights');
  $('model-description').textContent = `One shared E (97 × ${W}) embeds a and b and reads out the answer. The input is E[a] + Rotate90(E[b]): every adjacent pair (u, v) of b becomes (−v, u). This fixed rotation marks the operand role; it adds no learned weights. Both models have two identical-form residual blocks: RMSNorm → Neighbor mesh → Square → Neighbor mesh → Square → Neighbor mesh, plus skip. A mesh channel reads itself and the next channel, with circular indexing. Final RMSNorm and Eᵀ produce 97 logits. E's three roles share one AdamW state. The diagram shows one channel's 4 → 3 → 2 → 1 ancestors; RMSNorm uses all ${W} channels.`;
  $('reference-description').textContent = W === 1024 ? 'Default seed 11, Rust/CUDA: train ≥99% / held-out ≤10% at epochs 15–56 (41 epochs); held-out ≥90% at 136 and ≥99% at 187. All five tested seeds met the short-plateau rule: 41–42 plateau epochs, 74–80 rise epochs, held-out ≥99% at 175–190. Shading: train ≥99% in beige, perfect train in violet, held-out ≤10% in both. Browser arithmetic can change crossing epochs.' : 'Default seed 11, Rust/CUDA: held-out ≥90% at epoch 106 and ≥99% at 133. All five tested seeds reached held-out ≥99% at 123–137 epochs, without a memorization plateau. This is the same architecture as the wide model, with one eighth as many channels and parameters. Browser arithmetic can change crossing epochs.';
  $('criterion-description').textContent = neighbor ? 'At least 25 continuous epochs with train accuracy ≥99% and held-out accuracy ≤10%, followed by a ≤10% → ≥90% rise within 250 epochs and a later held-out ≥99% result. Every epoch is evaluated. The perfect-train interval is shown separately.' : 'Five consecutive observations with training accuracy ≥99% and held-out accuracy ≤10%, followed by held-out accuracy ≥99%.';
  $('scale-formula').textContent = '|probe hidden꜀| × ‖centered Eᵀ꜀‖: the magnitude of a channel’s readout contribution. If several channels share one screen pixel, its bar shows their maximum; clicking selects that channel. The channel input selects any channel exactly.';
  status(state.label); state.dirty = true;
}
function initialize() {
  if (!$('seed').reportValidity() || !$('limit').reportValidity()) return;
  const target = state;
  target.worker?.terminate();
  target.options = {model: target.model, seed: Number($('seed').value), backend: $('backend').value, maxEpochs: Number($('limit').value), captureHz: Number($('capture').value), stopAtTarget: $('stop-target').checked};
  target.seed = target.options.seed; target.initial = initialModelWeights(target.seed, target.model); target.weights = target.initial.slice();
  target.step = 0; target.activeMs = 0; target.history = []; target.criterion = null; target.ready = false; target.running = false; target.error = null; target.info = null;
  status('Loading', target); refresh();
  const worker = new Worker(new URL('./worker.mjs', import.meta.url), {type: 'module'}); target.worker = worker;
  worker.onmessage = ({data}) => {
    if (target.worker !== worker) return;
    if (data.type === 'ready') { target.ready = true; target.info = data.info; }
    else if (data.type === 'status') status(data.status, target);
    else if (data.type === 'weights') { target.weights = data.weights; target.step = data.step; target.activeMs = data.activeMs; }
    else if (data.type === 'metrics') { target.history.push(data.row); target.criterion = data.criterion; }
    else if (data.type === 'error') { status('Unavailable', target); target.ready = false; target.error = data.message; }
    if (state === target) refresh();
  };
  worker.onerror = event => { if (target.worker !== worker) return; status('Error', target); target.ready = false; target.error = event.message; if (state === target) refresh(); };
  worker.postMessage({type: 'initialize', options: target.options});
}
for (const button of document.querySelectorAll('[data-model-tab]')) button.onclick = () => {
  const id = button.dataset.modelTab; if (id === state.model) return;
  if (state.running) { state.worker.postMessage({type: 'pause'}); status('Pausing'); }
  const previous = state.options;
  state = experiments[id] ??= experiment(id);
  refresh();
  if (!state.worker) { state.options.backend = previous.backend; $('backend').value = previous.backend; initialize(); }
};
$('train').onclick = () => { if (state.ready) state.worker.postMessage({type: state.running ? 'pause' : 'train'}); };
$('step').onclick = () => state.worker.postMessage({type: 'step'});
$('reset').onclick = initialize;
$('seed').onchange = initialize;
$('backend').onchange = initialize;
for (const id of ['limit', 'stop-target']) $(id).onchange = () => {
  if (!$('limit').reportValidity()) return;
  const maxEpochs = Number($('limit').value);
  $('progress').max = maxEpochs; $('budget-label').textContent = `/ ${maxEpochs.toLocaleString('en-US')}`;
  state.options.maxEpochs = maxEpochs; state.options.stopAtTarget = $('stop-target').checked;
  state.worker?.postMessage({type: 'configure', maxEpochs, stopAtTarget: state.options.stopAtTarget});
};
$('capture').onchange = () => { state.options.captureHz = Number($('capture').value); state.worker?.postMessage({type: 'capture', hz: state.options.captureHz}); };
$('weight-group').onchange = () => { state.group = $('weight-group').value; refresh(); };
$('channel-select').oninput = () => {
  const channel = Number($('channel-select').value);
  if (Number.isInteger(channel) && channel >= 0 && channel < W) { state.channel = channel; state.dirty = true; }
};
for (const id of ['probe-a', 'probe-b', 'centered', 'weight-view']) $(id).oninput = () => { state.dirty = true; };
for (const [id, metric] of [['loss-tab', 'loss'], ['accuracy-tab', 'accuracy']]) $(id).onclick = () => {
  state.metric = metric; state.dirty = true;
  for (const [button, value] of [['loss-tab', 'loss'], ['accuracy-tab', 'accuracy']]) { $(button).classList.toggle('active', value === metric); $(button).setAttribute('aria-pressed', String(value === metric)); }
  $('axis-note').textContent = metric === 'loss' ? 'Cross-entropy · logarithmic scale' : 'Complete-split accuracy';
};
for (const id of ['input-a', 'input-b', 'readout', 'mesh', 'scales']) $(id).onclick = event => {
  if (id === 'mesh' && Boolean(modelInfo(state.model).meshDepth)) {
    const rect = $('mesh').getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
    const node = state.meshNodes?.find(node => Math.hypot(node.x - x, node.y - y) < 16);
    if (node) { state.channel = node.channel; $('channel-select').value = state.channel; state.dirty = true; }
    return;
  }
  const rect = $(id).getBoundingClientRect(), left = id === 'mesh' ? 76 : 0, right = id === 'mesh' ? 10 : 0;
  if (id === 'scales' && state.scaleBins?.length) {
    const index = Math.max(0, Math.min(state.scaleBins.length - 1, Math.floor((event.clientX - rect.left) / rect.width * state.scaleBins.length)));
    state.channel = state.scaleBins[index].channel; $('channel-select').value = state.channel; state.dirty = true; return;
  }
  const rowSelection = id === 'input-a' && atlasShape('a').columns !== W;
  state.channel = Math.max(0, Math.min(W - 1, Math.floor((rowSelection ? (event.clientY - rect.top) / rect.height : (event.clientX - rect.left - left) / (rect.width - left - right)) * W))); $('channel-select').value = state.channel; state.dirty = true;
};
for (const [id, type] of [['input-a', 'a'], ['input-b', 'b'], ['readout', 'r']]) $(id).onmousemove = event => {
  const {rows, columns} = atlasShape(type);
  const rect = $(id).getBoundingClientRect(), c = Math.max(0, Math.min(columns - 1, Math.floor((event.clientX - rect.left) / rect.width * columns))), k = Math.max(0, Math.min(rows - 1, Math.floor((event.clientY - rect.top) / rect.height * rows)));
  const index = atlasIndex(type, c, k);
  const value = atlasSign(type, c) * state.weights[index];
  $(id).title = Boolean(modelInfo(state.model).meshDepth) && state.group !== 'io' && type !== 'r' ? `Layer ${k + 1} · ${(c + Number(type === 'b')) % W} → ${c} · weight ${format(value)}` : `Channel ${c} · ${type === 'r' ? 'answer class' : 'residue'} ${k} · ${type === 'b' && modelInfo(state.model).rotaryEmbedding ? 'rotated E' : 'raw weight'} ${format(value)}`;
};
function download(content, name, type) { const url = URL.createObjectURL(new Blob([content], {type})), a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
$('export-weights').onclick = () => download(state.weights, `${state.model}-w${W}-seed${state.seed}-step${state.step}.f32`, 'application/octet-stream');
$('export-metrics').onclick = () => {
  const keys = ['step', 'epoch', 'trainLoss', 'heldoutLoss', 'trainAccuracy', 'heldoutAccuracy', 'phase'];
  download(keys.join(',') + '\n' + state.history.map(r => keys.map(k => r[k]).join(',')).join('\n') + '\n', `${state.model}-w${W}-seed${state.seed}-metrics.csv`, 'text/csv');
};
const resizeObserver = new ResizeObserver(() => { state.dirty = true; });
for (const element of document.querySelectorAll('main, canvas')) resizeObserver.observe(element);
for (const button of document.querySelectorAll('[data-view-tab]')) button.onclick = () => {
  document.body.dataset.view = button.dataset.viewTab;
  for (const tab of document.querySelectorAll('[data-view-tab]')) {
    const selected = tab === button; tab.classList.toggle('active', selected); tab.setAttribute('aria-pressed', String(selected));
  }
  state.dirty = true;
};
$('about').onclick = () => {
  $('engine-error').textContent = $('error').textContent;
  $('engine-error').hidden = $('error').hidden;
  $('about-dialog').showModal();
};
$('close-about').onclick = () => $('about-dialog').close();
$('error').onclick = () => $('about').click();
document.addEventListener('visibilitychange', () => { state.worker?.postMessage({type: 'capture', hz: document.hidden ? 1 : Number($('capture').value)}); state.dirty = true; });
document.fonts.ready.then(() => { state.dirty = true; });
initialize(); draw();
