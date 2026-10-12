struct Settings {
  rows: u32, offset: u32, bins: u32, mode: u32,
  lr: f32, decay: f32, correction1: f32, correction2: f32,
}
@group(0) @binding(0) var<uniform> s: Settings;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradients: array<f32>;
@group(0) @binding(3) var<storage, read_write> moments: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> data: array<u32>;
@group(0) @binding(5) var<storage, read> order: array<u32>;
@group(0) @binding(6) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(7) var<storage, read_write> errors: array<f32>;
@group(0) @binding(8) var<storage, read_write> statistics: array<vec2<f32>>;
const P: u32 = 97u;
const W: u32 = 64u;
const EMBEDDING: u32 = 6208u;
const PROJECTED: u32 = COUNT + 5u * W;
const DPROJECTED: u32 = PROJECTED + 2u * P * W;
const GAIN: u32 = 14400u;
const COUNT: u32 = 14656u;
const TOTAL: u32 = 9312u;
const TRAIN: u32 = 4656u;
const PLANE: u32 = TOTAL * W;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.98;

fn sample(row: u32) -> u32 {
  if s.mode == 1u { return order[s.offset + row]; }
  return row;
}

var<workgroup> activations: array<f32, 64>;
var<workgroup> values: array<f32, 128>;
var<workgroup> classes: array<u32, 128>;
var<workgroup> target_logit: f32;

// One workgroup owns a sample: cached shared-E projection, four own gains,
// dense readout, stable softmax, mean-loss derivative and sample statistics.
@compute @workgroup_size(128)
fn forward(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = group.x;
  let id = sample(row);
  let a = data[id * 3u]; let b = data[id * 3u + 1u]; let answer = data[id * 3u + 2u];
  if lane < W {
    let h = max(gradients[PROJECTED + a * W + lane] + gradients[PROJECTED + (P + b) * W + lane], 0.0) * gradients[COUNT + lane];
    activations[lane] = h;
    hidden[row * W + lane] = h;
  }
  workgroupBarrier();
  var logit = -3.402823e38;
  if lane < P {
    logit = 0.0;
    for (var c = 0u; c < W; c++) { logit += activations[c] * weights[lane * W + c]; }
    if lane == answer { target_logit = logit; }
  }
  values[lane] = logit; classes[lane] = lane;
  workgroupBarrier();
  for (var stride = 64u; stride > 0u; stride /= 2u) {
    if lane < stride {
      let other = values[lane + stride]; let other_class = classes[lane + stride];
      if other > values[lane] || (other == values[lane] && other_class < classes[lane]) {
        values[lane] = other; classes[lane] = other_class;
      }
    }
    workgroupBarrier();
  }
  let maximum = values[0]; let winner = classes[0];
  workgroupBarrier();
  var exponential = 0.0;
  if lane < P { exponential = exp(logit - maximum); }
  values[lane] = exponential;
  workgroupBarrier();
  for (var stride = 64u; stride > 0u; stride /= 2u) {
    if lane < stride { values[lane] += values[lane + stride]; }
    workgroupBarrier();
  }
  let denominator = values[0];
  if lane == 0u { statistics[row] = vec2<f32>(log(denominator) + maximum - target_logit, select(0.0, 1.0, winner == answer)); }
  if lane < P {
    if s.mode == 2u { errors[row * P + lane] = logit; }
    else { errors[row * P + lane] = (exponential / denominator - select(0.0, 1.0, lane == answer)) / f32(s.rows); }
  }
}

@compute @workgroup_size(64)
fn backward(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) c: u32) {
  let row = group.x; let id = sample(row);
  var dh = 0.0;
  for (var k = 0u; k < P; k++) { dh += errors[row * P + k] * weights[k * W + c]; }
  let input = gradients[PROJECTED + data[id * 3u] * W + c] + gradients[PROJECTED + (P + data[id * 3u + 1u]) * W + c];
  hidden[PLANE + row * W + c] = select(0.0, dh * gradients[COUNT + c], input > 0.0);
  hidden[2u * PLANE + row * W + c] = dh * max(input, 0.0);
}

var<workgroup> tile_h: array<f32, 64>;
var<workgroup> tile_e: array<f32, 64>;
// Tiled H^T dLogits: 8 channels by 8 classes, sharing minibatch reads.
@compute @workgroup_size(8, 8)
fn readout_gradient(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let c = group.y * 8u + local.y; let k = group.x * 8u + local.x;
  var result = 0.0;
  for (var start = 0u; start < s.rows; start += 8u) {
    var h = 0.0; var e = 0.0;
    if start + local.x < s.rows && c < W { h = hidden[(start + local.x) * W + c]; }
    if start + local.y < s.rows && k < P { e = errors[(start + local.y) * P + k]; }
    tile_h[local.y * 8u + local.x] = h; tile_e[local.y * 8u + local.x] = e;
    workgroupBarrier();
    for (var j = 0u; j < 8u; j++) { result += tile_h[local.y * 8u + j] * tile_e[j * 8u + local.x]; }
    workgroupBarrier();
  }
  if c < W && k < P { gradients[k * W + c] = result; }
}

@compute @workgroup_size(64)
fn input_gradient(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) c: u32) {
  let bin = group.x;
  var result = 0.0;
  for (var i = order[s.bins + bin]; i < order[s.bins + bin + 1u]; i++) {
    result += hidden[PLANE + order[i] * W + c];
  }
  gradients[DPROJECTED + bin * W + c] = result;
}

fn adam(parameter: u32, gradient: f32) {
  let m = BETA1 * moments[parameter].x + (1.0 - BETA1) * gradient;
  let v = BETA2 * moments[parameter].y + (1.0 - BETA2) * gradient * gradient;
  moments[parameter] = vec2<f32>(m, v);
  weights[parameter] -= s.lr * ((m / s.correction1) / (sqrt(v / s.correction2) + 1e-8) + s.decay * weights[parameter]);
}

@compute @workgroup_size(128)
fn dense_adam(@builtin(global_invocation_id) id: vec3<u32>) {
  if id.x < GAIN && s.lr > 0.0 { adam(id.x, gradients[id.x]); }
}

@compute @workgroup_size(64)
fn gain_adam(@builtin(local_invocation_index) c: u32) {
  var base = 0.0;
  for (var row = 0u; row < s.rows; row++) { base += hidden[2u * PLANE + row * W + c]; }
  // Read every derivative before replacing the cached factors.
  for (var layer = 0u; layer < 4u; layer++) {
    let index = GAIN + layer * W + c;
    let gradient = base * gradients[COUNT + W + layer * W + c];
    gradients[index] = gradient;
    if s.lr > 0.0 { adam(index, gradient); }
  }
  var u: array<f32, 4>;
  for (var layer = 0u; layer < 4u; layer++) { u[layer] = max(weights[GAIN + layer * W + c], 0.0); }
  gradients[COUNT + c] = u[0] * u[1] * u[2] * u[3];
  for (var layer = 0u; layer < 4u; layer++) {
    var derivative = select(0.0, 1.0, u[layer] > 0.0);
    for (var other = 0u; other < 4u; other++) { if other != layer { derivative *= u[other]; } }
    gradients[COUNT + W + layer * W + c] = derivative;
  }
}

var<workgroup> metric_loss: array<f32, 256>;
var<workgroup> metric_accuracy: array<f32, 256>;
@compute @workgroup_size(256)
fn reduce_statistics(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  var loss = 0.0; var accuracy = 0.0;
  for (var row = group.x * TRAIN + lane; row < (group.x + 1u) * TRAIN; row += 256u) {
    loss += statistics[row].x; accuracy += statistics[row].y;
  }
  metric_loss[lane] = loss; metric_accuracy[lane] = accuracy;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if lane < stride { metric_loss[lane] += metric_loss[lane + stride]; metric_accuracy[lane] += metric_accuracy[lane + stride]; }
    workgroupBarrier();
  }
  if lane == 0u { statistics[TOTAL + group.x] = vec2<f32>(metric_loss[0], metric_accuracy[0]) / f32(TRAIN); }
}

// Cache E A_a / E A_b once per optimizer update, independent of batch size.
@compute @workgroup_size(64)
fn project_inputs(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) c: u32) {
  let bin = group.x; let residue = bin % P; let operand = bin / P;
  var result = 0.0;
  for (var j = 0u; j < W; j++) { result += weights[residue * W + j] * weights[EMBEDDING + (operand * W + j) * W + c]; }
  gradients[PROJECTED + bin * W + c] = result;
}

// One invocation owns one unique parameter: deterministic gradient sums,
// including both input roles and the readout role of the shared embedding.
@compute @workgroup_size(128)
fn projection_gradient(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if index < EMBEDDING {
    let residue = index / W; let j = index % W; var result = gradients[index];
    for (var c = 0u; c < W; c++) {
      result += gradients[DPROJECTED + residue * W + c] * weights[EMBEDDING + j * W + c];
      result += gradients[DPROJECTED + (P + residue) * W + c] * weights[EMBEDDING + (W + j) * W + c];
    }
    gradients[index] = result;
  } else if index < GAIN {
    let at = index - EMBEDDING; let operand = at / (W * W); let j = (at / W) % W; let c = at % W;
    var result = 0.0;
    for (var residue = 0u; residue < P; residue++) { result += weights[residue * W + j] * gradients[DPROJECTED + (operand * P + residue) * W + c]; }
    gradients[index] = result;
  }
}
