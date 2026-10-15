import {element as $} from './dom.ts';
import type {CanvasId} from './dom.ts';
import {send, backend, preset} from './types.ts';
import type {Weights, ModelId, Preset, Metrics, Criterion, EngineInfo, TrainingOptions, TrainingCheckpoint, NetworkView, NetworkNode, NetworkEdge, Point, WeightMode, WorkerResponse} from './types.ts';
import type {Prediction} from './models.ts';
import {modelId, modelForWidth, trainingOptimizer} from './models.ts';
import {P} from './model.ts';
import {initializeI18n, t, number, runtimeText} from './i18n.ts';
import {initializeTask} from './task.ts';
import {pytorchSource} from './pytorch-export.ts';
import {modelInfo, modelPresets, optimizerProfile, initialModelWeights, predictModel, inputMatrix} from './models.ts';
import {WeightTopology, topologyLayout, weightGradient, weightMagnitude, weightColor, weightPalette, initialWeightRanges} from './topology.ts';

type AtlasType = 'a' | 'b' | 'r';
interface Experiment {
  model: ModelId; preset: Preset; weights: Weights; initial: Weights; history: Metrics[]; channel: number;
  meshView: 'topology' | 'channel'; networkMode: WeightMode; networkHighlight: boolean; networkView: NetworkView;
  metric: 'accuracy' | 'loss'; worker: Worker | null; running: boolean; ready: boolean; label: string;
  step: number; seed: number; activeMs: number; dirty: boolean; metricsDirty: boolean; options: TrainingOptions;
  projected?: Float64Array<ArrayBuffer>; projectedWeights?: Weights; criterion?: Criterion | null;
  probeCache?: {weights: Weights; a: number; b: number; result: Prediction};
  scalesCache?: {weights: Weights; result: Prediction | null; values: Float64Array};
  info?: EngineInfo | null; error?: string | null; focusWeight?: number | null;
  meshNodes?: NetworkNode[]; meshEdges?: NetworkEdge[];
  scaleBins?: {channel: number; start: number; end: number}[];
  checkpoint?: TrainingCheckpoint; suspension?: Promise<void>; finishSuspension?: () => void;
}
await initializeI18n();
function experiment(model: ModelId, preset: Preset): Experiment {
  const info = modelInfo(model), seed = info.seed, initial = initialModelWeights(seed, model);
  return {model, preset, weights: initial.slice(), initial, history: [], channel: 0, meshView: 'topology', networkMode: 'weight', networkHighlight: false, networkView: {zoom: 1, x: 0, y: 0}, metric: 'accuracy', worker: null, running: false, ready: false, label: 'Loading', step: 0, seed, activeMs: 0, dirty: true, metricsDirty: true,
    options: {model, preset, seed, backend: 'auto', maxEpochs: 10000, captureHz: 5}};
}
// Every width and optimizer preset keeps its own experiment.
const experiments = new Map<string, Experiment>([['dense-rotary/fast', experiment('dense-rotary', 'fast')]]);
let state = experiments.get('dense-rotary/fast')!, preferredPreset: Preset = 'fast';
let W = 128, activation = 0;
const renderInputs = new Map<CanvasId, readonly unknown[]>();
let pendingFrame: number | null = null, metricsTimer: number | null = null, lastMetricsPaint = 0;
const topology = new WeightTopology($('mesh-edges'), () => { renderInputs.delete('mesh'); invalidate(); });
for (const swatch of document.querySelectorAll<HTMLElement>('[data-weight-gradient]')) swatch.style.backgroundImage = weightGradient(swatch.dataset.weightGradient === 'positive');
const orange = '#dd6b3c', blue = '#49849c', ink = '#252d28', muted = '#96988e', line = '#e7e7dd';
const fixed = (value: number, digits: number) => number(value, {minimumFractionDigits: digits, maximumFractionDigits: digits, useGrouping: false});
const format = (value: number) => value === 0 ? '0' : Math.abs(value) < .001 ? value.toExponential(2) : fixed(value, 3);
function canvas(id: CanvasId) {
  const element = $(id), rect = element.getBoundingClientRect(), scale = Math.min(devicePixelRatio || 1, 2);
  const width = Math.round(rect.width), height = Math.round(rect.height);
  if (element.width !== Math.round(width * scale) || element.height !== Math.round(height * scale)) { element.width = Math.round(width * scale); element.height = Math.round(height * scale); }
  // Software rasterization keeps the node overlay independent of the driver.
  const ctx = element.getContext('2d', id === 'mesh' ? {willReadFrequently: true} : undefined);
  if (!ctx) throw new Error('Canvas 2D is unavailable.');
  ctx.setTransform(scale, 0, 0, scale, 0, 0); ctx.clearRect(0, 0, width, height);
  return {ctx, width, height};
}
function text(ctx: CanvasRenderingContext2D, label: string, x: number, y: number, align: CanvasTextAlign = 'left', colour = muted) { label = runtimeText(String(label)); ctx.font = '10px "DM Mono", monospace'; ctx.fillStyle = colour; ctx.textAlign = align; ctx.fillText(label, x, y); }
function curve() {
  const {ctx, width, height} = canvas('curve'), pad = {l: 48, r: 14, t: 18, b: 31};
  const rows = state.history, plotW = width - pad.l - pad.r, plotH = height - pad.t - pad.b;
  const interval = 50;
  const maxEpoch = Math.max(25, Math.ceil((rows[rows.length - 1]?.epoch || 0) / interval) * interval);
  const isLoss = state.metric === 'loss';
  const losses = rows.flatMap(r => [r.trainLoss, r.heldoutLoss]);
  const lo = isLoss ? Math.max(-8, Math.min(-1, Math.floor(Math.log10(Math.max(1e-8, Math.min(...losses, .1)))))) : 0;
  const hi = isLoss ? Math.max(1, Math.ceil(Math.log10(Math.max(...losses, 5)))) : 100;
  const x = (value: number) => pad.l + value / maxEpoch * plotW;
  const y = (value: number) => pad.t + plotH * (1 - ((isLoss ? Math.log10(Math.max(value, 1e-8)) : 100 * value) - lo) / (hi - lo));
  ctx.strokeStyle = line; ctx.lineWidth = 1;
  const ticks = isLoss ? Array.from({length: hi - lo + 1}, (_, i) => lo + i) : [0, 25, 50, 75, 100];
  for (const tick of ticks) {
    const py = pad.t + plotH * (1 - (tick - lo) / (hi - lo));
    ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(width - pad.r, py); ctx.stroke();
    text(ctx, isLoss ? `1e${tick}` : `${tick}%`, pad.l - 9, py + 3, 'right');
  }
  for (let i = 0; i <= 4; i++) text(ctx, number(Math.round(maxEpoch * i / 4)), x(maxEpoch * i / 4), height - 11, 'center');
  text(ctx, 'EPOCH', width - pad.r, 10, 'right');
  ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t - 3, plotW, plotH + 6); ctx.clip();
  for (const [band, colour] of [[state.criterion?.plateau, '#ddd7c8'], [state.criterion?.perfect, '#cbbad9']] as const) {
    if (band && band.start !== null && band.end !== null && band.end > band.start) { ctx.fillStyle = colour; ctx.globalAlpha = .4; ctx.fillRect(x(band.start), pad.t, x(band.end) - x(band.start), plotH); ctx.globalAlpha = 1; }
  }
  for (const [key, colour] of isLoss ? [['trainLoss', orange], ['heldoutLoss', blue]] as const : [['trainAccuracy', orange], ['heldoutAccuracy', blue]] as const) {
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
  if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || a >= P || b < 1 || b >= P) { ui('probe-result', 'Use integer a = 0…96 and b = 1…96.'); return null; }
  let cached = state.probeCache;
  if (!cached || cached.weights !== state.weights || cached.a !== a || cached.b !== b) {
    if (state.projectedWeights !== state.weights) { state.projected = inputMatrix(state.weights, state.model); state.projectedWeights = state.weights; }
    cached = {weights: state.weights, a, b, result: predictModel(state.weights, a, b, state.model, state.projected)};
    state.probeCache = cached;
  }
  const result = cached.result;
  let correct = 0; while (correct * b % P !== a) correct++;
  const maximum = Math.max(...result.logits), denominator = result.logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0);
  ui('prediction', result.answer);
  ui('probe-result', `Correct answer: ${correct} · confidence ${(100 / denominator).toFixed(1)}% · ${correct === result.answer ? 'correct' : 'not yet correct'}`);
  return result;
}
function rgb(value: number, range: number) {
  const t = Math.min(1, Math.abs(value) / Math.max(range, 1e-12));
  const target = value >= 0 ? [221, 107, 60] : [73, 132, 156];
  return target.map((c, i) => Math.round([249, 247, 239][i] * (1 - t) + c * t));
}
function denseMesh(result: Prediction | null, scales: Float64Array) {
  $('mesh-edges').hidden = state.meshView !== 'topology'; $('network-zoom').hidden = state.meshView !== 'topology'; $('network-options').hidden = state.meshView !== 'topology';
  return state.meshView === 'topology' ? denseTopology(result, scales) : denseChannel(result, scales);
}
function denseChannel(result: Prediction | null, scales: Float64Array) {
  const {ctx, width, height} = canvas('mesh'), model = modelInfo(state.model), stage = result?.stages[0], c = state.channel;
  const left = 35, right = width - 52, mid = (left + right) / 2, bottom = height - 34;
  const x = (i: number) => left + i / (W - 1) * (right - left), sourceY = 31, target = {x: mid, y: Math.max(70, height * .62), channel: c, value: Math.max(0, stage?.values[1][c] ?? 0), label: `ReLU(z)[${c}]`};
  const range = initialWeightRanges(state.initial, W)[1];
  const activationRange = Math.max(...Array.from(stage?.normalized.values ?? [], value => Math.abs(value)), 1e-9);
  state.meshNodes = []; state.meshEdges = [];
  for (let i = 0; i < W; i++) {
    const from = {x: x(i), y: sourceY, channel: i, value: stage?.normalized.values[i] ?? 0, label: `RMSNorm r[${i}]`}, weight = state.weights[model.gain + i * W + c];
    const magnitude = weightMagnitude(weight, range);
    ctx.beginPath(); ctx.strokeStyle = weightColor(weight >= 0, magnitude); ctx.globalAlpha = Math.max(.015, 1.5 / Math.sqrt(W)) * (.03 + .97 * magnitude); ctx.lineWidth = .6;
    ctx.moveTo(from.x, from.y); ctx.lineTo(target.x, target.y); ctx.stroke();
    state.meshNodes.push(from); state.meshEdges.push({from, to: target, weight, label: `A[${i},${c}]`});
  }
  ctx.globalAlpha = 1;
  for (const node of state.meshNodes) { ctx.beginPath(); ctx.fillStyle = colour(node.value ?? 0, activationRange); ctx.arc(node.x, node.y, Math.max(.6, Math.min(2.5, (right - left) / W * .25)), 0, 2 * Math.PI); ctx.fill(); }
  state.meshNodes.push(target); meshNode(ctx, target, Math.max(Math.abs(target.value), 1e-9));
  const sum = {x: mid, y: bottom}; meshArrow(ctx, target, sum, 1, 1, true);
  ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(right + 16, sourceY); ctx.lineTo(right + 16, bottom); ctx.lineTo(mid + 8, bottom); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.fillStyle = '#fffefa'; ctx.arc(sum.x, sum.y, 8, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); text(ctx, '+', sum.x, sum.y + 3, 'center', ink);
  text(ctx, 'RMSNorm r', mid, 12, 'center'); text(ctx, 'Dense → ReLU', mid, target.y - 13, 'center'); text(ctx, `skip x[${c}]`, right, bottom + 18, 'right');
  ui('selected-channel', 'Channel'); $('channel-select').value = String(c);
  ui('gain-info', `Block 1 · RMS ${format(stage?.normalized.inverse ?? 0)}`); ui('scale-info', `Contribution: ${format(scales[c])}`);
  ui('mesh-helper', `${W} → 1 · ${W} connections · Dense → ReLU + skip`);
  uiAttribute('mesh', 'aria-label', t('Dense residual block 1, channel {channel}, {width} to 1 neurons, {width} learned edges, pre-RMSNorm, ReLU and one skip connection', {channel: c, width: W}));
}
function meshArrow(ctx: CanvasRenderingContext2D, from: Point, to: Point, weight: number, range: number, selected: boolean) {
  const magnitude = Math.min(1, Math.abs(weight) / range), dx = to.x - from.x, dy = to.y - from.y, distance = Math.hypot(dx, dy);
  if (distance < 20) return;
  const ux = dx / distance, uy = dy / distance, end = {x: to.x - 9 * ux, y: to.y - 9 * uy};
  ctx.strokeStyle = ctx.fillStyle = weight === 0 ? '#d7d8cf' : weightColor(weight > 0, magnitude);
  ctx.globalAlpha = selected ? 1 : .65 + .35 * magnitude; ctx.lineWidth = .7 + 3 * magnitude;
  ctx.setLineDash(weight === 0 ? [2, 3] : []);
  ctx.beginPath(); ctx.moveTo(from.x + 8 * ux, from.y + 8 * uy); ctx.lineTo(end.x, end.y); ctx.stroke(); ctx.setLineDash([]);
  const length = 4 + 2 * magnitude, breadth = 2 + magnitude;
  ctx.beginPath(); ctx.moveTo(end.x, end.y);
  ctx.lineTo(end.x - length * ux + breadth * uy, end.y - length * uy - breadth * ux);
  ctx.lineTo(end.x - length * ux - breadth * uy, end.y - length * uy + breadth * ux); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
}
function meshNode(ctx: CanvasRenderingContext2D, node: NetworkNode, range: number) {
  ctx.beginPath(); ctx.fillStyle = '#fffefa'; ctx.arc(node.x, node.y, 6.5, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = node.channel === state.channel ? ink : '#989d96'; ctx.lineWidth = node.channel === state.channel ? 1.8 : 1.3; ctx.stroke();
  ctx.beginPath(); ctx.fillStyle = colour(node.value ?? 0, range); ctx.arc(node.x, node.y, 3, 0, 2 * Math.PI); ctx.fill();
}
function denseTopology(result: Prediction | null, scales: Float64Array) {
  const {ctx, width, height} = canvas('mesh'), model = modelInfo(state.model), layout = topologyLayout(width, height, W), view = state.networkView;
  const stage = result?.stages[0], rows = layout.rows, transform = (node: Point) => ({...node, x: (node.x - width / 2) * view.zoom + width / 2 + view.x, y: node.y + view.y});
  topology.draw(ctx, layout, state.weights, state.channel, view, state.focusWeight ?? -1, {initial: state.initial, mode: state.networkMode, highlight: state.networkHighlight});
  state.meshNodes = []; state.meshEdges = [];
  const hiddenRadius = Math.min(6.5, Math.max(.5, Math.min(3.5, (width - 40) / W * .25)) * Math.min(view.zoom, 5));
  const drawNode = (node: Point & {channel?: number; value?: number; label?: string}, radius: number) => {
    const n = transform(node); if (n.x < -12 || n.x > width + 12 || n.y < -12 || n.y > height + 12) return;
    // Keep node styling stable while learned connections change.
    ctx.beginPath(); ctx.fillStyle = '#fffefa'; ctx.arc(n.x, n.y, radius, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = '#969d94'; ctx.lineWidth = Math.min(1, radius * .45); ctx.stroke();
    if (typeof node.channel === 'number') state.meshNodes!.push({...n, ...node, x: n.x, y: n.y, channel: node.channel});
  };
  for (const [layer, values, label] of [[1, stage?.values[0], 'RMSNorm r'], [2, result?.hidden, 'Final RMSNorm h']] as const) {
    for (let c = 0; c < W; c++) drawNode({x: layout.column(c), y: rows[layer], channel: c, value: values?.[c], label: `${label}[${c}]`}, hiddenRadius);
  }
  const tokenRadius = Math.min(6.5, Math.min(2.6, (width - 64) / (4 * P)) * Math.min(view.zoom, 4));
  for (let r = 0; r < P; r++) {
    for (const role of [0, 1]) drawNode({x: layout.residue(r, role), y: rows[0]}, tokenRadius);
    drawNode({x: layout.answer(r), y: rows[3]}, tokenRadius);
  }
  // The skip carries x0, before the first normalization; the layer contains
  // both x0 and r. Norms use the complete vector and add no learned weights.
  if (state.networkHighlight) {
    const start = transform({x: layout.column(state.channel), y: rows[1]}), end = transform({x: layout.column(state.channel), y: rows[2]});
    ctx.strokeStyle = ink; ctx.lineWidth = .8; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(start.x + 5, start.y);
    ctx.bezierCurveTo(start.x + 16, start.y, end.x + 16, end.y, end.x + 5, end.y); ctx.stroke(); ctx.setLineDash([]);
  }
  const label = (value: string, x: number, y: number) => { value = runtimeText(value); const n = transform({x, y}); ctx.font = '10px "DM Mono", monospace'; ctx.textAlign = 'center'; const length = ctx.measureText(value).width;
    ctx.fillStyle = '#fffefae8'; ctx.fillRect(n.x - length / 2 - 3, n.y - 10, length + 6, 14); ctx.fillStyle = ink; ctx.fillText(value, n.x, n.y); };
  label('E · a', layout.residue(48, 0), rows[0] - 10); label('J(E) · b', layout.residue(48, 1), rows[0] - 10);
  label('x₀ → RMSNorm', width / 2, rows[1] - 9);
  label('Dense → ReLU + skip → RMSNorm', width / 2, rows[2] - 9);
  label('Tied Eᵀ · 97 answers', width / 2, rows[3] + 16);
  const connections = Number($('mesh-edges').dataset.connections);
  ui('mesh-helper', `${number(connections)} connections · ${number(model.parameters)} parameters`);
  ui('mesh-title', state.networkMode === 'delta' ? 'All weight changes' : 'All weights');
  $('channel-select').value = String(state.channel);
  ui('gain-info', `CPU drawing · ${view.zoom.toFixed(1)}× · fixed scale`);
  ui('scale-info', `Contribution: ${format(scales[state.channel])}`);
  uiAttribute('mesh', 'aria-label', runtimeText(`All model ${state.networkMode === 'delta' ? 'weight changes from initialization' : 'weights'}, channel ${state.channel}, ${connections} connections from ${model.parameters} unique parameters, shared E in a, Rotate90 b and tied readout roles, ${W} channels, one Dense matrix, pre-RMSNorm, ReLU and residual skip, fixed color scale${state.networkHighlight ? ', selected channel highlighted' : ', no channel emphasis'}`));
}
function colour(value: number, range: number) { return `rgb(${rgb(value, range).join(',')})`; }
const ATLAS_COLOR_LEVELS = 1024, atlasPalette = weightPalette(ATLAS_COLOR_LEVELS);
const atlasCache = new Map<AtlasType, {weights: Weights; initial: Weights; mode: string; centered: boolean; bitmap: HTMLCanvasElement; image: ImageData; maximum: number}>();
function atlasShape(type: AtlasType) {return {rows: type === 'b' ? W : P, columns: W};}
function atlasIndex(type: AtlasType, channel: number, row: number) {return type === 'b' ? modelInfo(state.model).gain + row * W + channel : row * W + channel;}
function atlas(id: CanvasId, type: AtlasType) {
  const {rows, columns} = atlasShape(type);
  const {ctx, width, height} = canvas(id), mode = $('weight-view').value, centered = type === 'r' && $('centered').checked;
  const range = initialWeightRanges(state.initial, W)[type === 'b' ? 1 : 0] * (mode === 'delta' ? .25 : 1);
  let cached = atlasCache.get(type);
  if (!cached || cached.weights !== state.weights || cached.initial !== state.initial || cached.mode !== mode || cached.centered !== centered) {
    const bitmap = cached?.bitmap ?? document.createElement('canvas');
    const resized = bitmap.width !== columns || bitmap.height !== rows;
    if (resized) { bitmap.width = columns; bitmap.height = rows; }
    const bitmapCtx = bitmap.getContext('2d')!, image = !cached || resized ? bitmapCtx.createImageData(columns, rows) : cached.image;
    const weights = state.weights, original = mode === 'delta' ? state.initial : null, offset = type === 'b' ? modelInfo(state.model).gain : 0;
    const means = centered ? new Float64Array(columns) : null;
    if (means) {
      for (let k = 0; k < P; k++) for (let c = 0; c < columns; c++) {
        const index = k * columns + c;
        means[c] += weights[index] - (original ? original[index] : 0);
      }
      for (let c = 0; c < columns; c++) means[c] /= P;
    }
    // Scan contiguous weights once and reuse the pixel buffer between snapshots.
    // Preserve the former f32 color rounding without a million-element scratch array.
    let maximum = 0;
    const colorRange = Math.max(range, 1e-9);
    for (let i = 0; i < rows * columns; i++) {
      const index = offset + i, raw = weights[index] - (original ? original[index] : 0) - (means ? means[i % columns] : 0);
      maximum = Math.max(maximum, Math.abs(raw));
      const value = Math.fround(raw), at = i * 4;
      if (value === 0) {
        image.data[at] = 249; image.data[at + 1] = 247; image.data[at + 2] = 239;
      } else {
        const strength = Math.log1p(Math.abs(value) / colorRange), shade = Math.round(strength / (1 + strength) * (ATLAS_COLOR_LEVELS - 1));
        const color = ((value > 0 ? 0 : ATLAS_COLOR_LEVELS) + shade) * 4;
        image.data[at] = atlasPalette[color]; image.data[at + 1] = atlasPalette[color + 1]; image.data[at + 2] = atlasPalette[color + 2];
      }
      image.data[at + 3] = 255;
    }
    bitmapCtx.putImageData(image, 0, 0);
    cached = {weights: state.weights, initial: state.initial, mode, centered, bitmap, image, maximum}; atlasCache.set(type, cached);
  }
  ctx.imageSmoothingEnabled = false; ctx.drawImage(cached.bitmap, 0, 0, width, height);
  ctx.strokeStyle = ink; ctx.lineWidth = 1;
  if (columns !== W) ctx.strokeRect(.5, state.channel * height / rows + .5, width - 1, height / rows - 1);
  else ctx.strokeRect(state.channel * width / W, .5, Math.max(1, width / W - 1), height - 1);
  $(id).dataset.colorScale = 'logarithmic'; $(id).dataset.colorReference = String(range);
  $(id).dataset.colorLevels = String(ATLAS_COLOR_LEVELS);
  ui('range-' + type, `max ${format(cached.maximum)}`);
}
function channelScales(result: Prediction | null) {
  const cached = state.scalesCache;
  if (cached?.weights === state.weights && cached.result === result) return cached.values;
  const scales = new Float64Array(W);
  for (let c = 0; c < W; c++) {
    let norm = 0, mean = 0;
    for (let k = 0; k < P; k++) mean += state.weights[k * W + c] / P;
    for (let k = 0; k < P; k++) norm += (state.weights[k * W + c] - mean) ** 2;
    scales[c] = Math.abs(result?.hidden[c] ?? 0) * Math.sqrt(norm);
  }
  state.scalesCache = {weights: state.weights, result, values: scales};
  return scales;
}
function scalePlot(values: Float64Array) {
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
  ui('concentration', `Top 16: ${(100 * top / Math.max(total, 1e-12)).toFixed(1)}%`);
}
function invalidate(target = state, immediate = true) {
  target.dirty = true;
  if (target !== state || document.hidden || pendingFrame !== null) return;
  if (immediate) {
    if (metricsTimer !== null) { clearTimeout(metricsTimer); metricsTimer = null; }
    pendingFrame = requestAnimationFrame(draw);
  } else if (metricsTimer === null) {
    // Every epoch stays in history; UI-only progress paints at most ten times a second.
    metricsTimer = window.setTimeout(() => {
      metricsTimer = null;
      if (!document.hidden && pendingFrame === null) pendingFrame = requestAnimationFrame(draw);
    }, Math.max(0, 100 - (performance.now() - lastMetricsPaint)));
  }
}
function renderCanvas(id: CanvasId, inputs: readonly unknown[], render: () => void) {
  const rect = $(id).getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const next = [...inputs, Math.round(rect.width), Math.round(rect.height), Math.min(devicePixelRatio || 1, 2), document.documentElement.lang];
  const previous = renderInputs.get(id);
  if (previous && previous.length === next.length && next.every((value, i) => value === previous[i])) return;
  render(); renderInputs.set(id, next);
}
function draw() {
  pendingFrame = null;
  if (!state.dirty || document.hidden) return;
  state.dirty = false;
  if (state.metricsDirty) refreshMetrics();
  renderCanvas('curve', [state, state.history.length, state.metric, state.criterion], curve);
  const visible = (id: string) => { const rect = $(id).getBoundingClientRect(); return rect.width > 0 && rect.height > 0; };
  // The atlas needs no forward pass. Pan, selection and repaint reuse predictions.
  const meshVisible = visible('mesh'), scalesVisible = visible('scales');
  if (visible('probe-a') || meshVisible || scalesVisible) {
    const result = probe();
    if (meshVisible || scalesVisible) {
      const scales = channelScales(result), view = state.networkView;
      renderCanvas('mesh', [state, state.weights, state.initial, result, state.channel, state.meshView, state.networkMode, state.networkHighlight, state.networkHighlight ? state.focusWeight : null, view.zoom, view.x, view.y], () => denseMesh(result, scales));
      renderCanvas('scales', [state, scales, state.channel], () => scalePlot(scales));
    }
  }
  for (const [id, type] of [['input-a', 'a'], ['input-b', 'b'], ['readout', 'r']] as const) {
    renderCanvas(id, [state, state.weights, state.initial, state.channel, $('weight-view').value, type === 'r' && $('centered').checked], () => atlas(id, type));
  }
}
function status(label: string, target = state) {
  target.label = label; target.running = label === 'Training' || label === 'Pausing';
  if (target !== state) return;
  ui('status', label);
  const labelText = t(state.running ? 'Pause training' : state.step > 0 ? 'Continue training' : 'Start training') + ' ';
  if ($('train').firstChild?.textContent !== labelText) {
    const icon = document.createElement('span'); icon.textContent = state.running ? 'Ⅱ' : '↗';
    $('train').replaceChildren(document.createTextNode(labelText), icon);
  }
  $('train').classList.toggle('running', state.running);
  $('train').disabled = !state.ready || label === 'Pausing';
  $('step').disabled = !state.ready || state.running;
  for (const id of ['seed', 'limit', 'backend'] as const) $(id).disabled = state.running;
}
function refreshMetrics() {
  const options = state.options, row = state.history.at(-1), progressStep = Math.max(state.step, row?.step ?? 0);
  const epoch = Math.floor(progressStep / 10) + (progressStep % 10) * 512 / 4656;
  ui('epoch', number(epoch, {maximumFractionDigits: 1}));
  $('progress').value = epoch; $('progress').max = options.maxEpochs;
  ui('budget-label', '/ ' + number(options.maxEpochs));
  ui('observations', state.history.length + ' observations');
  ui('phase', row?.phase ?? 'Waiting for the first observation');
  for (const [id, key] of [['train-accuracy', 'trainAccuracy'], ['heldout-accuracy', 'heldoutAccuracy']] as const) ui(id, row ? fixed(100 * row[key], 2) + '%' : '—');
  for (const [id, key] of [['train-loss', 'trainLoss'], ['heldout-loss', 'heldoutLoss']] as const) ui(id, row ? format(row[key]) : '—');
  ui('throughput', state.activeMs ? number(Math.round(state.step * 1000 / state.activeMs)) + ' updates / sec' : state.ready ? 'Ready when you are' : 'Preparing training engine');
  const seconds = Math.floor(state.activeMs / 1000); ui('elapsed', Math.floor(seconds / 60) + ':' + String(seconds % 60).padStart(2, '0'));
  state.metricsDirty = false; lastMetricsPaint = performance.now();
}
function refresh() {
  const options = state.options, model = modelInfo(state.model);
  W = model.width;
  $('seed').value = String(options.seed); $('limit').value = String(options.maxEpochs); $('backend').value = options.backend;
  $('capture').value = String(options.captureHz);
  ui('model-name', model.title + ' · ' + t(model.subtitle));
  uiTitle('model-name', $('model-name').textContent);
  ui('model-detail', W + ' channels · ' + number(model.parameters) + ' parameters · ' + t('ReLU') + ' · FP32');
  ui('cadence', 'Evaluated every epoch');
  ui('step', '+1 epoch');
  ui('atlas-r-size', W + ' × 97');
  $('channel-select').max = String(W - 1); $('channel-select').value = String(state.channel);
  refreshMetrics();
  const info = state.info;
  ui('device', info ? `${runtimeText(info.description.split(' · ').slice(0, 2).join(' · '))} · ${t('{count} threads', {count: info.threads})}` : state.error ? 'Training engine could not start' : 'Preparing training engine…');
  uiTitle('device', $('device').textContent);
  ui('backend-note', info?.note ?? ''); $('backend-note').hidden = !info?.note;
  $('error').hidden = !state.error; ui('error', state.error ?? ''); uiTitle('error', state.error ?? '');
  for (const [id, metric] of [['loss-tab', 'loss'], ['accuracy-tab', 'accuracy']] as const) { $(id).classList.toggle('active', state.metric === metric); $(id).setAttribute('aria-pressed', String(state.metric === metric)); }
  for (const [id, view] of [['mesh-topology', 'topology'], ['mesh-channel', 'channel']] as const) { $(id).classList.toggle('active', state.meshView === view); $(id).setAttribute('aria-pressed', String(state.meshView === view)); }
  $('network-view').value = state.networkMode; $('network-highlight').checked = state.networkHighlight;
  ui('axis-note', state.metric === 'loss' ? 'Cross-entropy · logarithmic scale' : 'Complete-split accuracy');
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-model-tab]')) {
    const active = !options.optimizer && button.dataset.modelTab === state.model; button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
  }
  $('custom-tab').hidden = !experiments.has('custom');
  $('custom-tab').classList.toggle('active', Boolean(options.optimizer));
  $('custom-tab').setAttribute('aria-pressed', String(Boolean(options.optimizer)));
  const presets = modelPresets(state.model);
  $('preset-nav').hidden = Boolean(options.optimizer) || presets.length < 2;
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-preset-tab]')) {
    const active = button.dataset.presetTab === state.preset; button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
    button.hidden = !presets.includes(preset(button.dataset.presetTab ?? ''));
  }
  ui('atlas-a-title', 'Shared embedding E'); ui('atlas-b-title', 'Dense matrix A'); ui('atlas-r-title', 'Tied readout Eᵀ');
  for (const type of ['a', 'b'] as const) {const shape = atlasShape(type); ui('atlas-' + type + '-size', shape.rows + ' × ' + shape.columns);}
  ui('atlas-a-axis', 'Residue ↓ · channel →'); ui('atlas-b-axis', 'Input channel ↓ · output channel →');
  ui('mesh-title', state.meshView === 'topology' ? state.networkMode === 'delta' ? 'All weight changes' : 'All weights' : 'Residual block 1');
  ui('scales-title', 'Probe contributions');
  $('scales').setAttribute('aria-label', t('Probe readout contributions for all {width} channels', {width: W}));
  for (const type of ['a', 'b'] as const) $('input-' + type).setAttribute('aria-label', runtimeText($('atlas-' + type + '-title').textContent + ' weights'));
  ui('model-description', t('One shared E (97 × {width}) embeds a and b and reads out the answer. Input: E[a] + Rotate90(E[b]). One pre-norm residual block applies RMSNorm → Dense A → ReLU, then adds the original input. Final RMSNorm and Eᵀ give 97 logits. E starts normal with variance 1/{width}; A is Kaiming normal with variance 2/{width}; there are no biases or learned norm gains.', {width: W}));
  const profile = options.optimizer ? null : optimizerProfile(state.model, state.preset);
  const optimizer = trainingOptimizer(state.model, state.preset, options.optimizer);
  const reference = profile ? t('Rust/CUDA, seeds {seeds}: epochs to ≥99% held-out accuracy: {epochs}. A dash means not reached within {budget} epochs. Browser crossings may differ.', {
    seeds: profile.nativeSeeds.join(' / '), epochs: profile.nativeTest99Epochs.map(epoch => epoch === null ? '—' : number(epoch)).join(' / '), budget: number(profile.nativeBudget),
  }) : t('Custom experiment');
  const plateau = profile?.nativePlateauEpochs ? ' ' + t('Memorization plateau: {epochs} epochs.', {epochs: profile.nativePlateauEpochs.map(epochs => number(epochs)).join(' / ')}) : '';
  ui('reference-description', `AdamW · lr=${optimizer.learningRate} · wd=${optimizer.weightDecay} · β=0.9/0.98 · ε=1e-8 · warmup=10. ${reference}${plateau}`);
  $('reference-description').dataset.learningRate = String(optimizer.learningRate); $('reference-description').dataset.weightDecay = String(optimizer.weightDecay);
  ui('model-optimizer', `lr=${optimizer.learningRate} · wd=${optimizer.weightDecay}`);
  ui('criterion-description', t('At least 25 continuous epochs with train ≥99% and held-out ≤10%, then a rise to ≥90% within 250 epochs and a later held-out ≥99% result. Every epoch is evaluated.'));
  ui('scale-formula', t('|probe hidden꜀| × ‖centered Eᵀ꜀‖ measures a channel’s readout contribution. If channels share a pixel, its bar shows their maximum and selects that channel. The channel input selects any channel exactly.'));
  status(state.label); invalidate();
}
function releaseWorker(target: Experiment) {
  target.worker?.terminate(); target.worker = null;
  const finish = target.finishSuspension;
  target.suspension = undefined; target.finishSuspension = undefined; finish?.();
}
function suspendExperiment(target: Experiment): Promise<void> {
  if (target.suspension) return target.suspension;
  if (!target.worker) return Promise.resolve();
  target.ready = false;
  target.suspension = new Promise(resolve => {target.finishSuspension = resolve;});
  send(target.worker, {type: 'suspend'});
  return target.suspension;
}
function initialize() {
  if (!$('seed').reportValidity() || !$('limit').reportValidity()) return;
  activation++;
  releaseWorker(state); state.checkpoint = undefined;
  startEngine();
}
function startEngine(checkpoint?: TrainingCheckpoint) {
  const target = state;
  target.options = {...target.options, model: target.model, preset: target.preset, seed: Number($('seed').value), backend: backend($('backend').value), maxEpochs: Number($('limit').value), captureHz: Number($('capture').value)};
  if (!checkpoint) {
    target.seed = target.options.seed; target.initial = initialModelWeights(target.seed, target.model); target.weights = target.initial.slice();
    target.step = 0; target.activeMs = 0; target.history = []; target.criterion = null;
  }
  target.ready = false; target.running = false; target.error = null; target.info = null;
  status('Loading', target); refresh();
  const worker = new Worker(new URL('./worker.js', import.meta.url), {type: 'module'}); target.worker = worker;
  worker.onmessage = ({data}: MessageEvent<WorkerResponse>) => {
    if (target.worker !== worker) return;
    if (data.type === 'suspended') {
      target.checkpoint = data.checkpoint; target.ready = false; target.running = false;
      releaseWorker(target); return;
    }
    if (data.type === 'ready') { target.ready = true; target.info = data.info; }
    else if (data.type === 'status') status(data.status, target);
    else if (data.type === 'weights') { target.weights = data.weights; target.step = data.step; target.activeMs = data.activeMs; }
    else if (data.type === 'metrics') { target.history.push(data.row); target.criterion = data.criterion; }
    else if (data.type === 'error') { status('Unavailable', target); target.ready = false; target.error = data.message; }
    target.metricsDirty = true;
    if (state === target && (data.type === 'ready' || data.type === 'status' || data.type === 'error')) refresh();
    else invalidate(target, data.type !== 'metrics');
  };
  worker.onerror = event => { if (target.worker !== worker) return; status('Error', target); target.ready = false; target.error = event.message; releaseWorker(target); if (state === target) refresh(); };
  const transfer = checkpoint ? [checkpoint.engine.weights.buffer, checkpoint.engine.moment1.buffer, checkpoint.engine.moment2.buffer, checkpoint.sampler.order.buffer] : [];
  worker.postMessage({type: 'initialize', options: target.options, checkpoint}, transfer);
  target.checkpoint = undefined;
}
function select(model: ModelId, chosen: Preset) {
  if (!state.options.optimizer && model === state.model && chosen === state.preset) return;
  const key = `${model}/${chosen}`;
  if (!experiments.has(key)) experiments.set(key, experiment(model, chosen));
  void activate(experiments.get(key)!);
}
async function activate(target: Experiment) {
  if (target === state) return;
  const request = ++activation, previous = state.options;
  void suspendExperiment(state);
  state = target;
  if (!target.worker && !target.checkpoint && !target.history.length) target.options.backend = previous.backend;
  refresh();
  // Finish the previous worker before allocating another engine or thread pool.
  await Promise.all([...experiments.values()].filter(experiment => experiment !== target || experiment.suspension).map(suspendExperiment));
  if (request !== activation || state !== target) return;
  if (!target.worker) startEngine(target.checkpoint);
}
// Widths without the preferred preset open their fast preset.
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-model-tab]')) button.onclick = () => {
  const id = modelId(button.dataset.modelTab ?? '');
  select(id, modelPresets(id).includes(preferredPreset) ? preferredPreset : 'fast');
};
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-preset-tab]')) button.onclick = () => {
  preferredPreset = preset(button.dataset.presetTab ?? ''); select(state.model, preferredPreset);
};
$('custom-tab').onclick = () => {const custom = experiments.get('custom'); if (custom) void activate(custom);};
let settingsDraft = false;
function summarizeSettings() {
  if (!$('custom-width').checkValidity()) {ui('custom-summary', ''); return;}
  const info = modelInfo(modelForWidth(Number($('custom-width').value)));
  ui('custom-summary', t('{width} channels · {parameters} parameters · {activation} · FP32', {width: info.width, parameters: number(info.parameters), activation: t('ReLU')}));
}
for (const id of ['custom-width', 'custom-lr', 'custom-wd'] as const) $(id).oninput = () => {
  settingsDraft = true; $('custom-error').hidden = true; summarizeSettings();
};
$('settings-form').onsubmit = event => {
  event.preventDefault();
  if (!$('settings-form').reportValidity()) return;
  try {
    const model = modelForWidth(Number($('custom-width').value));
    const optimizer = trainingOptimizer(model, 'fast', {learningRate: Number($('custom-lr').value), weightDecay: Number($('custom-wd').value)});
    const custom = experiment(model, 'fast');
    custom.options = {...state.options, model, preset: 'fast', optimizer};
    const previous = experiments.get('custom');
    if (previous) {releaseWorker(previous); previous.running = false;}
    experiments.set('custom', custom); void activate(custom);
    $('custom-error').hidden = true; selectView('training');
  } catch (error) {
    ui('custom-error', error instanceof Error ? error.message : String(error)); $('custom-error').hidden = false;
  }
};
$('train').onclick = () => { if (state.ready) send(state.worker, {type: state.running ? 'pause' : 'train'}); };
$('step').onclick = () => send(state.worker, {type: 'step'});
$('reset').onclick = initialize;
$('seed').onchange = initialize;
$('backend').onchange = initialize;
$('limit').onchange = () => {
  if (!$('limit').reportValidity()) return;
  const maxEpochs = Number($('limit').value);
  $('progress').max = maxEpochs; ui('budget-label', `/ ${number(maxEpochs)}`);
  state.options.maxEpochs = maxEpochs;
  send(state.worker, {type: 'configure', maxEpochs});
};
$('capture').onchange = () => { state.options.captureHz = Number($('capture').value); send(state.worker, {type: 'capture', hz: state.options.captureHz}); };
$('channel-select').oninput = () => {
  const channel = Number($('channel-select').value);
  if (Number.isInteger(channel) && channel >= 0 && channel < W && channel !== state.channel) { state.channel = channel; invalidate(); }
};
$('network-view').onchange = () => { state.networkMode = $('network-view').value === 'delta' ? 'delta' : 'weight'; invalidate(); };
$('network-highlight').onchange = () => { state.networkHighlight = $('network-highlight').checked; invalidate(); };
for (const [id, view] of [['mesh-topology', 'topology'], ['mesh-channel', 'channel']] as const) $(id).onclick = () => { state.meshView = view; state.focusWeight = null; refresh(); };
function zoomNetwork(factor: number, x?: number, y?: number) {
  const rect = $('mesh').getBoundingClientRect(), view = state.networkView, zoom = Math.max(1, Math.min(64, view.zoom * factor)), ratio = zoom / view.zoom;
  x ??= rect.width / 2; y ??= rect.height / 2;
  view.x = x - rect.width / 2 - (x - rect.width / 2 - view.x) * ratio;
  view.zoom = zoom; if (zoom === 1) view.x = view.y = 0;
  invalidate();
}
$('zoom-in').onclick = () => zoomNetwork(2);
$('zoom-out').onclick = () => zoomNetwork(.5);
$('zoom-reset').onclick = () => { state.networkView = {zoom: 1, x: 0, y: 0}; invalidate(); };
$('mesh').onwheel = event => { if (state.meshView !== 'topology') return; event.preventDefault(); const rect = $('mesh').getBoundingClientRect(); zoomNetwork(Math.exp(-event.deltaY * .002), event.clientX - rect.left, event.clientY - rect.top); };
$('mesh').onkeydown = event => {
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); state.channel = (state.channel + (event.key === 'ArrowLeft' ? W - 1 : 1)) % W; invalidate(); }
  else if (event.key === '+' || event.key === '=') { event.preventDefault(); zoomNetwork(2); }
  else if (event.key === '-') { event.preventDefault(); zoomNetwork(.5); }
  else if (event.key === '0') { event.preventDefault(); $('zoom-reset').click(); }
};
let networkDrag: (Point & {panX: number; panY: number}) | null = null, suppressMeshClick = false;
$('mesh').onpointerdown = event => {
  if (state.meshView !== 'topology') return;
  networkDrag = {x: event.clientX, y: event.clientY, panX: state.networkView.x, panY: state.networkView.y};
  suppressMeshClick = false; $('mesh').setPointerCapture(event.pointerId);
};
$('mesh').onpointermove = event => {
  if (networkDrag) {
    const dx = event.clientX - networkDrag.x, dy = event.clientY - networkDrag.y;
    if (Math.hypot(dx, dy) > 3) suppressMeshClick = true;
    if (suppressMeshClick) { state.networkView.x = networkDrag.panX + dx; invalidate(); }
    return;
  }
  const rect = $('mesh').getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
  const nearest = state.meshNodes?.reduce<NetworkNode | null>((best, node) => !best || Math.hypot(node.x - x, node.y - y) < Math.hypot(best.x - x, best.y - y) ? node : best, null);
  uiTitle('mesh', nearest && Math.hypot(nearest.x - x, nearest.y - y) < 10 ? `${nearest.label ?? 'Channel ' + nearest.channel} · activation ${format(nearest.value ?? 0)} · click to select` : 'All: wheel to zoom, drag to pan. Arrow keys select a channel; 0 fits the network.');
};
for (const type of ['pointerup', 'pointercancel'] as const) $('mesh').addEventListener(type, () => { networkDrag = null; });
for (const id of ['probe-a', 'probe-b', 'centered', 'weight-view'] as const) $(id).oninput = () => { invalidate(); };
for (const [id, metric] of [['loss-tab', 'loss'], ['accuracy-tab', 'accuracy']] as const) $(id).onclick = () => {
  state.metric = metric; invalidate();
  for (const [button, value] of [['loss-tab', 'loss'], ['accuracy-tab', 'accuracy']] as const) { $(button).classList.toggle('active', value === metric); $(button).setAttribute('aria-pressed', String(value === metric)); }
  ui('axis-note', metric === 'loss' ? 'Cross-entropy · logarithmic scale' : 'Complete-split accuracy');
};
for (const id of ['input-a', 'input-b', 'readout', 'mesh', 'scales'] as const) $(id).onclick = event => {
  if (id === 'mesh') {
    if (suppressMeshClick) { suppressMeshClick = false; return; }
    const rect = $('mesh').getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
    const node = state.meshNodes?.reduce<NetworkNode | null>((best, node) => !best || Math.hypot(node.x - x, node.y - y) < Math.hypot(best.x - x, best.y - y) ? node : best, null);
    if (node && Math.hypot(node.x - x, node.y - y) < 16) { state.channel = node.channel; $('channel-select').value = String(state.channel); invalidate(); }
    return;
  }
  const rect = $(id).getBoundingClientRect();
  if (id === 'scales' && state.scaleBins?.length) {
    const index = Math.max(0, Math.min(state.scaleBins.length - 1, Math.floor((event.clientX - rect.left) / rect.width * state.scaleBins.length)));
    state.channel = state.scaleBins[index].channel; $('channel-select').value = String(state.channel); invalidate(); return;
  }
  const rowSelection = id === 'input-a' && atlasShape('a').columns !== W;
  state.channel = Math.max(0, Math.min(W - 1, Math.floor((rowSelection ? (event.clientY - rect.top) / rect.height : (event.clientX - rect.left) / rect.width) * W))); $('channel-select').value = String(state.channel); invalidate();
};
for (const [id, type] of [['input-a', 'a'], ['input-b', 'b'], ['readout', 'r']] as const) $(id).onmousemove = event => {
  const {rows, columns} = atlasShape(type);
  const rect = $(id).getBoundingClientRect(), c = Math.max(0, Math.min(columns - 1, Math.floor((event.clientX - rect.left) / rect.width * columns))), k = Math.max(0, Math.min(rows - 1, Math.floor((event.clientY - rect.top) / rect.height * rows)));
  const index = atlasIndex(type, c, k);
  const value = state.weights[index];
  if (state.focusWeight !== index) {
    state.focusWeight = index;
    if (document.body.dataset.view === 'channels' && state.networkHighlight) invalidate();
  }
  uiTitle(id, type === 'b' ? `A[${k},${c}] · ${k} → ${c} · weight ${format(value)}` : `Channel ${c} · ${type === 'r' ? 'answer class' : 'residue'} ${k} · raw weight ${format(value)}`);
};
for (const id of ['input-a', 'input-b', 'readout'] as const) $(id).onmouseleave = () => {
  if (state.focusWeight === null) return;
  state.focusWeight = null;
  if (document.body.dataset.view === 'channels' && state.networkHighlight) invalidate();
};
function download(content: BlobPart, name: string, type: string) { const url = URL.createObjectURL(new Blob([content], {type})), a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
const presetSuffix = () => state.options.optimizer ? `-custom-lr${state.options.optimizer.learningRate}-wd${state.options.optimizer.weightDecay}` : state.preset === 'fast' ? '' : '-' + state.preset;
$('export-pytorch').onclick = () => download(pytorchSource(state.options), `rotate90-w${W}${presetSuffix()}-seed${state.seed}.py`, 'text/x-python');
$('export-metrics').onclick = () => {
  const keys = ['step', 'epoch', 'trainLoss', 'heldoutLoss', 'trainAccuracy', 'heldoutAccuracy', 'phase'] as const;
  download(keys.join(',') + '\n' + state.history.map(r => keys.map(k => r[k]).join(',')).join('\n') + '\n', `${state.model}${presetSuffix()}-w${W}-seed${state.seed}-metrics.csv`, 'text/csv');
};
const resizeObserver = new ResizeObserver(() => { invalidate(); });
for (const element of document.querySelectorAll('main, canvas')) resizeObserver.observe(element);
window.addEventListener('resize', () => { invalidate(); });
function selectView(view: string) {
  document.body.dataset.view = view;
  $('settings-panel').hidden = view !== 'settings';
  if (view === 'settings') {
    if (!settingsDraft) {
      const optimizer = trainingOptimizer(state.model, state.preset, state.options.optimizer);
      $('custom-width').value = String(W); $('custom-lr').value = String(optimizer.learningRate); $('custom-wd').value = String(optimizer.weightDecay);
    }
    summarizeSettings();
  }
  for (const tab of document.querySelectorAll<HTMLButtonElement>('[data-view-tab]')) {
    const selected = tab.dataset.viewTab === view; tab.classList.toggle('active', selected); tab.setAttribute('aria-pressed', String(selected));
  }
  invalidate();
}
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-view-tab]')) button.onclick = () => selectView(button.dataset.viewTab!);
$('about').onclick = () => {
  ui('engine-error', $('error').textContent);
  $('engine-error').hidden = $('error').hidden;
  $('about-dialog').showModal();
};
$('close-about').onclick = () => $('about-dialog').close();
$('error').onclick = () => $('about').click();
document.addEventListener('visibilitychange', () => { send(state.worker, {type: 'capture', hz: document.hidden ? 1 : Number($('capture').value)}); invalidate(); });
document.fonts.ready.then(() => { invalidate(); });
window.addEventListener('localechange', () => { refresh(); summarizeSettings(); });
initialize(); initializeTask(); invalidate();

function ui(id: string, value: string | number) {
  const element = $(id), next = runtimeText(String(value));
  if (element.textContent !== next) element.textContent = next;
}
function uiAttribute(id: string, name: string, value: string) {
  const element = $(id);
  if (element.getAttribute(name) !== value) element.setAttribute(name, value);
}
function uiTitle(id: string, value: string | null) { uiAttribute(id, 'title', runtimeText(String(value))); }
