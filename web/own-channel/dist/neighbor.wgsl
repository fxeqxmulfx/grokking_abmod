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
const W: u32 = 128u;
const READOUT: u32 = 24832u;
const GAIN: u32 = 37248u;
const COUNT: u32 = 38784u;
const TOTAL: u32 = 9312u;
const TRAIN: u32 = 4656u;
// Cache only a minibatch. Full-split evaluation uses workgroup-local states.
const PLANE: u32 = 512u * W;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.98;

fn sample(row: u32) -> u32 {
  if s.mode == 1u { return order[s.offset + row]; }
  return row;
}

var<workgroup> activations: array<f32, 128>;
var<workgroup> skip: array<f32, 128>;
var<workgroup> derivative: array<f32, 128>;
var<workgroup> values: array<f32, 128>;
var<workgroup> classes: array<u32, 128>;
var<workgroup> sums: array<vec2<f32>, 128>;
var<workgroup> inverse: f32;
var<workgroup> normalized_dot: f32;
var<workgroup> target_logit: f32;

// Planes: states 0/4/8; normalized inputs 1/5/9; pre-Square values 2/3/6/7;
// six mesh output derivatives 10..15; operand projection derivative 16.
fn reduce_norm(lane: u32) {
  workgroupBarrier();
  for (var stride = 64u; stride > 0u; stride /= 2u) {
    if lane < stride { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  if lane == 0u { inverse = inverseSqrt(sums[0].x / f32(W) + 1e-5); normalized_dot = sums[0].y / f32(W); }
  workgroupBarrier();
}

fn normalize(lane: u32) {
  sums[lane] = vec2<f32>(activations[lane] * activations[lane], 0.0);
  reduce_norm(lane);
  activations[lane] *= inverse;
  workgroupBarrier();
}

fn norm_backward(row: u32, state: u32, norm: u32, lane: u32) {
  let x = hidden[state * PLANE + row * W + lane];
  let n = hidden[norm * PLANE + row * W + lane];
  sums[lane] = vec2<f32>(x * x, n * derivative[lane]);
  reduce_norm(lane);
  derivative[lane] = inverse * (derivative[lane] - n * normalized_dot);
  workgroupBarrier();
}

@compute @workgroup_size(128)
fn forward(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = group.x; let id = sample(row);
  let a = data[id * 3u]; let b = data[id * 3u + 1u]; let answer = data[id * 3u + 2u];
  activations[lane] = weights[a * W + lane] + weights[(P + b) * W + lane];
  if s.mode == 1u { hidden[row * W + lane] = activations[lane]; }
  workgroupBarrier();
  for (var block = 0u; block < 2u; block++) {
    skip[lane] = activations[lane];
    normalize(lane);
    if s.mode == 1u { hidden[(block * 4u + 1u) * PLANE + row * W + lane] = activations[lane]; }
    for (var layer = 0u; layer < 3u; layer++) {
      let at = GAIN + ((block * 3u + layer) * W + lane) * 2u;
      var own = activations[lane]; var other = activations[(lane + 1u) % W];
      if layer > 0u { own *= own; other *= other; }
      let next = own * weights[at] + other * weights[at + 1u];
      workgroupBarrier();
      activations[lane] = next;
      if s.mode == 1u && layer < 2u { hidden[(block * 4u + layer + 2u) * PLANE + row * W + lane] = next; }
      workgroupBarrier();
    }
    activations[lane] += skip[lane];
    if s.mode == 1u { hidden[(block * 4u + 4u) * PLANE + row * W + lane] = activations[lane]; }
    workgroupBarrier();
  }
  normalize(lane);
  if s.mode == 1u { hidden[9u * PLANE + row * W + lane] = activations[lane]; }
  var logit = -3.402823e38;
  if lane < P {
    logit = 0.0;
    for (var c = 0u; c < W; c++) { logit += activations[c] * weights[READOUT + c * P + lane]; }
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
  var dh = 0.0;
  for (var k = 0u; k < P; k++) { dh += errors[row * P + k] * weights[READOUT + lane * P + k]; }
  derivative[lane] = dh;
  workgroupBarrier();
  norm_backward(row, 8u, 9u, lane);
  for (var remaining = 2u; remaining > 0u; remaining--) {
    let block = remaining - 1u; let skip_gradient = derivative[lane];
    for (var remaining_layer = 3u; remaining_layer > 0u; remaining_layer--) {
      let layer = remaining_layer - 1u; let flat = block * 3u + layer;
      let previous = (lane + W - 1u) % W;
      hidden[(10u + flat) * PLANE + row * W + lane] = derivative[lane];
      var dx = weights[GAIN + (flat * W + lane) * 2u] * derivative[lane]
             + weights[GAIN + (flat * W + previous) * 2u + 1u] * derivative[previous];
      if layer > 0u { dx *= 2.0 * hidden[(block * 4u + layer + 1u) * PLANE + row * W + lane]; }
      workgroupBarrier();
      derivative[lane] = dx;
      workgroupBarrier();
    }
    norm_backward(row, block * 4u, block * 4u + 1u, lane);
    derivative[lane] += skip_gradient;
    workgroupBarrier();
  }
  hidden[16u * PLANE + row * W + lane] = derivative[lane];
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
    if start + local.x < s.rows && c < W { h = hidden[9u * PLANE + (start + local.x) * W + c]; }
    if start + local.y < s.rows && k < P { e = errors[(start + local.y) * P + k]; }
    tile_h[local.y * 8u + local.x] = h; tile_e[local.y * 8u + local.x] = e;
    workgroupBarrier();
    for (var j = 0u; j < 8u; j++) { result += tile_h[local.y * 8u + j] * tile_e[j * 8u + local.x]; }
    workgroupBarrier();
  }
  if c < W && k < P { gradients[READOUT + c * P + k] = result; }
}

@compute @workgroup_size(128)
fn input_gradient(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) c: u32) {
  let bin = group.x;
  var result = 0.0;
  for (var i = order[s.bins + bin]; i < order[s.bins + bin + 1u]; i++) {
    result += hidden[16u * PLANE + order[i] * W + c];
  }
  gradients[bin * W + c] = result;
}



// Each thread owns both incoming edge gradients for one neuron. No float atomics.
@compute @workgroup_size(128)
fn mesh_gradient(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) c: u32) {
  let flat = group.x; let block = flat / 3u; let layer = flat % 3u;
  let source_plane = block * 4u + layer + 1u; let delta_plane = 10u + flat;
  var own_gradient = 0.0; var neighbor_gradient = 0.0;
  for (var row = 0u; row < s.rows; row++) {
    var own = hidden[source_plane * PLANE + row * W + c];
    var other = hidden[source_plane * PLANE + row * W + (c + 1u) % W];
    if layer > 0u { own *= own; other *= other; }
    let delta = hidden[delta_plane * PLANE + row * W + c];
    own_gradient += own * delta; neighbor_gradient += other * delta;
  }
  let at = GAIN + (flat * W + c) * 2u;
  gradients[at] = own_gradient; gradients[at + 1u] = neighbor_gradient;
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
