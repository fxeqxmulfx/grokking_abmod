export const P = 97, W = 64, LAYERS = 4;
export const INPUT = 2 * P * W, READOUT = P * W;
export const GAIN = INPUT + READOUT, PARAMETERS = GAIN + LAYERS * W;
export const TRAIN = P * (P - 1) / 2, TOTAL = TRAIN * 2;

// SplitMix64 using pairs of u32s. No BigInt allocations in the epoch sampler.
function highMultiply(a, b) {
  const a0 = a & 65535, a1 = a >>> 16, b0 = b & 65535, b1 = b >>> 16;
  const t = a1 * b0 + ((a0 * b0) >>> 16);
  return (a1 * b1 + (t >>> 16) + (((t & 65535) + a0 * b1) >>> 16)) >>> 0;
}

export class Random {
  constructor(seed) {
    this.low = Number(seed & 0xffffffffn);
    this.high = Number((seed >> 32n) & 0xffffffffn);
    this.resultHigh = 0;
  }
  next() {
    const old = this.low;
    this.low = (old + 0x7f4a7c15) >>> 0;
    this.high = (this.high + 0x9e3779b9 + Number(this.low < old)) >>> 0;
    let a = (this.low ^ (this.low >>> 30 | this.high << 2)) >>> 0;
    let b = (this.high ^ this.high >>> 30) >>> 0;
    b = (highMultiply(a, 0x1ce4e5b9) + Math.imul(b, 0x1ce4e5b9) + Math.imul(a, 0xbf58476d)) >>> 0;
    a = Math.imul(a, 0x1ce4e5b9) >>> 0;
    a = (a ^ (a >>> 27 | b << 5)) >>> 0;
    b = (b ^ b >>> 27) >>> 0;
    b = (highMultiply(a, 0x133111eb) + Math.imul(b, 0x133111eb) + Math.imul(a, 0x94d049bb)) >>> 0;
    a = Math.imul(a, 0x133111eb) >>> 0;
    this.resultHigh = (b ^ b >>> 31) >>> 0;
    return (a ^ (a >>> 31 | b << 1)) >>> 0;
  }
  unit() { this.next(); return (this.resultHigh >>> 8) / 16777216; }
  index(bound) {
    const threshold = ((4294967296 % bound) ** 2) % bound;
    let low;
    do { low = this.next(); } while (this.resultHigh === 0 && low < threshold);
    return ((this.resultHigh % bound) * 4294967296 + low) % bound;
  }
  shuffle(values) {
    for (let end = values.length - 1; end > 0; end--) {
      const j = this.index(end + 1);
      const v = values[end]; values[end] = values[j]; values[j] = v;
    }
  }
}

export function initialWeights(seed = 7) {
  const random = new Random(BigInt(seed) ^ 0x73696d706c650004n);
  const weights = new Float32Array(PARAMETERS);
  for (const [start, count, fanIn] of [[0, INPUT, 2 * P], [INPUT, READOUT, W]]) {
    const bound = Math.fround(1 / Math.fround(Math.sqrt(fanIn)));
    for (let i = start; i < start + count; i++) weights[i] = Math.fround((2 * random.unit() - 1) * bound);
  }
  weights.fill(1, GAIN);
  return weights;
}

export function dataset(seed = 7) {
  const rows = [];
  for (let a = 0; a < P; a++) for (let b = 1; b < P; b++) {
    let inverse = 1;
    while (inverse * b % P !== 1) inverse++;
    rows.push([a, b, a * inverse % P]);
  }
  new Random(BigInt(seed) ^ 0x73706c6974730002n).shuffle(rows);
  return Uint32Array.from(rows.flat());
}

export class Sampler {
  constructor(seed = 7) {
    this.random = new Random(BigInt(seed) ^ 0x6261746368657303n);
    this.order = Uint32Array.from({length: TRAIN}, (_, i) => i);
    this.random.shuffle(this.order);
    this.position = 0;
  }
  sample(batch = 512) {
    if (this.position === TRAIN) { this.random.shuffle(this.order); this.position = 0; }
    const start = this.position;
    this.position = Math.min(TRAIN, start + batch);
    return this.order.subarray(start, this.position);
  }
}

// Sparse input-gradient reduction: deterministic lists of minibatch rows for
// each one-hot operand. This avoids float atomics and scanning 512 rows per weight.
export function packBatch(data, sample, buffer, offset = 0) {
  const n = sample.length, bins = offset + n, list = bins + 2 * P + 1;
  buffer.set(sample, offset);
  const counts = new Uint32Array(2 * P);
  for (const id of sample) { counts[data[id * 3]]++; counts[P + data[id * 3 + 1]]++; }
  let cursor = list;
  for (let bin = 0; bin < 2 * P; bin++) { buffer[bins + bin] = cursor; cursor += counts[bin]; }
  buffer[bins + 2 * P] = cursor;
  const positions = buffer.slice(bins, bins + 2 * P);
  for (let row = 0; row < n; row++) {
    const id = sample[row];
    buffer[positions[data[id * 3]]++] = row;
    buffer[positions[P + data[id * 3 + 1]]++] = row;
  }
  return {rows: n, offset, bins, end: cursor};
}

export function predict(weights, a, b) {
  const inputHidden = new Float64Array(W), hidden = new Float64Array(W), logits = new Float64Array(P);
  for (let c = 0; c < W; c++) {
    let h = Math.max(0, weights[a * W + c] + weights[(P + b) * W + c]);
    inputHidden[c] = h;
    for (let l = 0; l < LAYERS; l++) h = Math.max(0, h * weights[GAIN + l * W + c]);
    hidden[c] = h;
    for (let k = 0; k < P; k++) logits[k] += h * weights[INPUT + c * P + k];
  }
  const answer = logits.indexOf(Math.max(...logits));
  return {answer, inputHidden, hidden, logits};
}

export class Grokking {
  constructor() { this.streak = 0; this.plateau = false; this.onset = null; this.confirmed = null; this.event = null; }
  observe(row) {
    if (row.trainAccuracy >= .99 && row.heldoutAccuracy <= .1) {
      if (!this.streak && !this.plateau) this.onset = row.epoch;
      this.streak++;
      if (this.streak === 5 && !this.plateau) { this.plateau = true; this.confirmed = row.epoch; }
    } else this.streak = 0;
    if (row.heldoutAccuracy >= .99 && this.event === null) {
      this.event = {epoch: row.epoch, delayed: this.plateau && this.confirmed < row.epoch};
    }
    return this.event?.delayed ? 'Delayed grokking' : this.event ? 'Generalized' : this.plateau ? 'Memorization plateau observed' : 'Learning';
  }
}

// The one-epoch rule used in the Rust Neighbor study. Freeze its first 90%
// crossing; later dips cannot turn a slow rise into a sharp one.
export class ShortGrokking {
  constructor() {
    this.plateau = {start: null, end: null, epochs: 0};
    this.perfect = {start: null, end: null, epochs: 0};
    this.start = null; this.perfectStart = null; this.previous = null;
    this.lastLow = null; this.first90 = null; this.rise = null; this.event = null;
  }
  observe(row) {
    if (this.previous !== null && Math.abs(row.epoch - this.previous - 1) > 1e-6) this.start = this.perfectStart = null;
    this.previous = row.epoch;
    if (this.first90 === null) {
      const low = row.heldoutAccuracy <= .1;
      this.start = low && row.trainAccuracy >= .99 ? this.start ?? row.epoch : null;
      this.perfectStart = low && row.trainAccuracy === 1 ? this.perfectStart ?? row.epoch : null;
      for (const [start, name] of [[this.start, 'plateau'], [this.perfectStart, 'perfect']]) {
        if (start !== null && (this[name].start === null || row.epoch - start > this[name].epochs)) this[name] = {start, end: row.epoch, epochs: row.epoch - start};
      }
      if (low) this.lastLow = row.epoch;
      if (row.heldoutAccuracy >= .9) {
        this.first90 = row.epoch;
        this.rise = this.lastLow === null ? null : row.epoch - this.lastLow;
      }
    }
    if (this.event === null && row.heldoutAccuracy >= .99) this.event = {
      epoch: row.epoch, delayed: this.plateau.epochs >= 25 && this.rise !== null && this.rise <= 250,
      plateauEpochs: this.plateau.epochs, perfectTrainEpochs: this.perfect.epochs, riseEpochs: this.rise,
    };
    return this.event?.delayed ? 'Short-plateau grokking' : this.event ? 'Generalized' : this.plateau.epochs >= 25 && row.heldoutAccuracy <= .1 ? 'Memorization plateau observed' : row.heldoutAccuracy > .1 ? 'Generalizing' : 'Learning';
  }
}
