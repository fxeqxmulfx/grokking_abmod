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
const INPUT: u32 = 77952u;
const GAIN: u32 = 12416u;
const COUNT: u32 = 84160u;
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

const X: u32 = 256u;
// Planes 0..2: residual states, 3..5: normalized states, 6..13: expanded
// activations, 14..15: branch upstream, 16..17: normalized derivatives,
// 18..25: expansion derivatives, 26: input derivative.
var<workgroup> expansion: array<f32, 256>;
var<workgroup> derivative: array<f32, 64>;
var<workgroup> expansion_derivative: array<f32, 256>;
var<workgroup> reduction: array<f32, 128>;
var<workgroup> inverse: f32;
var<workgroup> normalized_dot: f32;

fn normalize(lane: u32) {
  reduction[lane] = 0.0;
  if lane < W { reduction[lane] = activations[lane] * activations[lane]; }
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if lane < stride { reduction[lane] += reduction[lane + stride]; }
    workgroupBarrier();
  }
  if lane == 0u { inverse = inverseSqrt(reduction[0] / f32(W) + 1e-5); }
  workgroupBarrier();
  if lane < W { activations[lane] *= inverse; }
  workgroupBarrier();
}

fn norm_backward(row: u32, block: u32, lane: u32) {
  if lane == 0u {
    var square = 0.0; var dot = 0.0;
    for (var c = 0u; c < W; c++) {
      let h = hidden[block * PLANE + row * W + c];
      square += h * h;
      dot += hidden[(3u + block) * PLANE + row * W + c] * derivative[c];
    }
    inverse = inverseSqrt(square / f32(W) + 1e-5); normalized_dot = dot / f32(W);
  }
  workgroupBarrier();
  if lane < W { derivative[lane] = inverse * (derivative[lane] - hidden[(3u + block) * PLANE + row * W + lane] * normalized_dot); }
  workgroupBarrier();
}

@compute @workgroup_size(128)
fn forward(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = group.x; let id = sample(row);
  let a = data[id * 3u]; let b = data[id * 3u + 1u]; let answer = data[id * 3u + 2u];
  if lane < W { activations[lane] = weights[a * W + lane] + weights[(P + b) * W + lane]; hidden[row * W + lane] = activations[lane]; }
  workgroupBarrier();
  for (var block = 0u; block < 2u; block++) {
    normalize(lane);
    if lane < W { hidden[(3u + block) * PLANE + row * W + lane] = activations[lane]; }
    let at = GAIN + block * 2u * W * X;
    for (var j = lane; j < X; j += 128u) {
      var value = 0.0;
      for (var c = 0u; c < W; c++) { value += activations[c] * weights[at + c * X + j]; }
      expansion[j] = max(0.0, value);
      hidden[(6u + block * 4u) * PLANE + row * X + j] = expansion[j];
    }
    workgroupBarrier();
    if lane < W {
      var branch = 0.0;
      for (var j = 0u; j < X; j++) { branch += expansion[j] * weights[at + W * X + j * W + lane]; }
      activations[lane] = hidden[block * PLANE + row * W + lane] + branch;
      hidden[(block + 1u) * PLANE + row * W + lane] = activations[lane];
    }
    workgroupBarrier();
  }
  normalize(lane);
  if lane < W { hidden[5u * PLANE + row * W + lane] = activations[lane]; }
  var logit = -3.402823e38;
  if lane < P {
    logit = 0.0;
    for (var c = 0u; c < W; c++) { logit += activations[c] * weights[INPUT + c * P + lane]; }
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


@compute @workgroup_size(128)
fn backward(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = group.x;
  if lane < W {
    var dh = 0.0;
    for (var k = 0u; k < P; k++) { dh += errors[row * P + k] * weights[INPUT + lane * P + k]; }
    derivative[lane] = dh;
  }
  workgroupBarrier();
  norm_backward(row, 2u, lane);
  for (var remaining = 2u; remaining > 0u; remaining--) {
    let block = remaining - 1u; let at = GAIN + block * 2u * W * X;
    if lane < W { hidden[(14u + block) * PLANE + row * W + lane] = derivative[lane]; }
    for (var j = lane; j < X; j += 128u) {
      var dx = 0.0;
      for (var c = 0u; c < W; c++) { dx += derivative[c] * weights[at + W * X + j * W + c]; }
      expansion_derivative[j] = select(0.0, dx, hidden[(6u + block * 4u) * PLANE + row * X + j] > 0.0);
      hidden[(18u + block * 4u) * PLANE + row * X + j] = expansion_derivative[j];
    }
    workgroupBarrier();
    if lane < W {
      var dn = 0.0;
      for (var j = 0u; j < X; j++) { dn += expansion_derivative[j] * weights[at + lane * X + j]; }
      derivative[lane] = dn;
      hidden[(16u + block) * PLANE + row * W + lane] = dn;
    }
    workgroupBarrier();
    norm_backward(row, block, lane);
    if lane < W { derivative[lane] += hidden[(14u + block) * PLANE + row * W + lane]; }
    workgroupBarrier();
  }
  if lane < W { hidden[26u * PLANE + row * W + lane] = derivative[lane]; }
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
    if start + local.x < s.rows && c < W { h = hidden[5u * PLANE + (start + local.x) * W + c]; }
    if start + local.y < s.rows && k < P { e = errors[(start + local.y) * P + k]; }
    tile_h[local.y * 8u + local.x] = h; tile_e[local.y * 8u + local.x] = e;
    workgroupBarrier();
    for (var j = 0u; j < 8u; j++) { result += tile_h[local.y * 8u + j] * tile_e[j * 8u + local.x]; }
    workgroupBarrier();
  }
  if c < W && k < P { gradients[INPUT + c * P + k] = result; }
}

@compute @workgroup_size(64)
fn input_gradient(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) c: u32) {
  let bin = group.x;
  var result = 0.0;
  for (var i = order[s.bins + bin]; i < order[s.bins + bin + 1u]; i++) {
    result += hidden[26u * PLANE + order[i] * W + c];
  }
  gradients[bin * W + c] = result;
}


// Four tiled dense derivatives: expansion and contraction in each block.
@compute @workgroup_size(8, 8)
fn matrix_gradient(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let block = group.z / 2u; let contraction = group.z % 2u == 1u;
  let inputs = select(W, X, contraction); let outputs = select(X, W, contraction);
  if group.y * 8u >= inputs || group.x * 8u >= outputs { return; }
  let i = group.y * 8u + local.y; let j = group.x * 8u + local.x;
  let xplane = select(3u + block, 6u + 4u * block, contraction);
  let dplane = select(18u + 4u * block, 14u + block, contraction);
  var result = 0.0;
  for (var start = 0u; start < s.rows; start += 8u) {
    var h = 0.0; var d = 0.0;
    if start + local.x < s.rows { h = hidden[xplane * PLANE + (start + local.x) * inputs + i]; }
    if start + local.y < s.rows { d = hidden[dplane * PLANE + (start + local.y) * outputs + j]; }
    tile_h[local.y * 8u + local.x] = h; tile_e[local.y * 8u + local.x] = d;
    workgroupBarrier();
    for (var k = 0u; k < 8u; k++) { result += tile_h[local.y * 8u + k] * tile_e[k * 8u + local.x]; }
    workgroupBarrier();
  }
  let at = GAIN + block * 2u * W * X + select(0u, W * X, contraction);
  gradients[at + i * outputs + j] = result;
}
fn adam(parameter: u32, gradient: f32) {
  let m = BETA1 * moments[parameter].x + (1.0 - BETA1) * gradient;
  let v = BETA2 * moments[parameter].y + (1.0 - BETA2) * gradient * gradient;
  moments[parameter] = vec2<f32>(m, v);
  weights[parameter] -= s.lr * ((m / s.correction1) / (sqrt(v / s.correction2) + 1e-8) + s.decay * weights[parameter]);
}

@compute @workgroup_size(128)
fn dense_adam(@builtin(global_invocation_id) id: vec3<u32>) {
  if id.x < COUNT && s.lr > 0.0 { adam(id.x, gradients[id.x]); }
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
