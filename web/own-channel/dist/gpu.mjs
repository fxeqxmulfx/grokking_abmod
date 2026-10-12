import {P, TOTAL, TRAIN, dataset, packBatch} from './model.mjs';
import {modelInfo, initialModelWeights} from './models.mjs';

const SLOTS = 32, STRIDE = 256, BATCH_WORDS = 3 * 512 + 2 * P + 1;
const STAGES = ['forward', 'backward', 'readout_gradient', 'input_gradient', 'dense_adam', 'gain_adam', 'reduce_statistics'];

export class WebGPUTrainer {
  static async create(seed = 7, options = {}) {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable. Open this page in a WebGPU browser over HTTPS or localhost.');
    const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter) throw new Error('No WebGPU adapter is available. Check that hardware acceleration is enabled.');
    const device = await adapter.requestDevice();
    const engine = new WebGPUTrainer(device, adapter, seed, options);
    try { await engine.initialize(options.weights ?? initialModelWeights(seed, engine.model.id)); }
    catch (error) { device.destroy(); throw error; }
    return engine;
  }
  constructor(device, adapter, seed, options) {
    this.device = device; this.adapter = adapter; this.data = dataset(seed);
    this.model = modelInfo(options.model ?? 'original');
    this.step = 0; this.b1 = 1; this.b2 = 1;
    this.learningRate = options.learningRate ?? .001;
    this.decay = options.decay ?? .1;
    this.buffers = [];
    this.failure = null;
    device.addEventListener('uncapturederror', event => { this.failure = event.error; });
    device.lost.then(info => { if (info.reason !== 'destroyed') this.failure = new Error(`GPU connection lost: ${info.message}`); });
  }
  buffer(label, size, usage) {
    const buffer = this.device.createBuffer({label, size, usage});
    this.buffers.push(buffer); return buffer;
  }
  async initialize(weights) {
    const {parameters: PARAMETERS, gain: GAIN} = this.model;
    const W = this.model.width, neighbor = Boolean(this.model.meshDepth);
    if (weights.length !== PARAMETERS || !weights.every(Number.isFinite)) throw new Error('Invalid model weights.');
    const device = this.device, U = GPUBufferUsage;
    const shader = await fetch(new URL(`./${this.model.id === 'original' ? 'train' : this.model.id}.wgsl`, import.meta.url));
    if (!shader.ok) throw new Error(`Could not load training shader: HTTP ${shader.status}`);
    const module = device.createShaderModule({label: this.model.title, code: await shader.text()});
    const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (messages.length) throw new Error(messages.map(m => `WGSL ${m.lineNum}:${m.linePos}: ${m.message}`).join('\n'));
    device.pushErrorScope('validation');
    this.uniforms = this.buffer('Step uniforms', SLOTS * STRIDE, U.UNIFORM | U.COPY_DST);
    this.weights = this.buffer('Weights', PARAMETERS * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    this.gradients = this.buffer('Gradients and cached projections', (PARAMETERS + 5 * W + (this.model.projection !== undefined ? 4 * P * W : 0)) * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    this.moments = this.buffer('Independent AdamW moments', PARAMETERS * 8, U.STORAGE);
    this.examples = this.buffer('Complete seeded division table', this.data.byteLength, U.STORAGE | U.COPY_DST);
    this.order = this.buffer('Sparse batches', SLOTS * BATCH_WORDS * 4, U.STORAGE | U.COPY_DST);
    this.hidden = this.buffer('Hidden activations and derivatives', (neighbor ? 512 * 17 : TOTAL * (this.model.id === 'residual' ? 27 : 3)) * W * 4, U.STORAGE);
    this.errors = this.buffer('Softmax derivatives', TOTAL * P * 4, U.STORAGE | U.COPY_SRC);
    this.statistics = this.buffer('Per-equation metrics', (TOTAL + 2) * 8, U.STORAGE | U.COPY_SRC);
    this.weightReadback = this.buffer('Weight readback', PARAMETERS * 4, U.COPY_DST | U.MAP_READ);
    this.metricReadback = this.buffer('Metric readback', 16, U.COPY_DST | U.MAP_READ);
    device.queue.writeBuffer(this.weights, 0, weights);
    device.queue.writeBuffer(this.examples, 0, this.data);
    const factors = new Float32Array(5 * W);
    for (let c = 0; c < W && this.model.gains; c++) {
      const gains = Array.from({length: 4}, (_, l) => Math.max(0, weights[GAIN + l * W + c]));
      factors[c] = gains.reduce((a, b) => a * b, 1);
      for (let l = 0; l < 4; l++) factors[W + l * W + c] = gains[l] > 0 ? gains.reduce((a, b, i) => a * (i === l ? 1 : b), 1) : 0;
    }
    device.queue.writeBuffer(this.gradients, PARAMETERS * 4, factors);
    const buffers = [this.uniforms, this.weights, this.gradients, this.moments, this.examples, this.order, this.hidden, this.errors, this.statistics];
    const layout = device.createBindGroupLayout({entries: buffers.map((_, binding) => ({binding, visibility: GPUShaderStage.COMPUTE,
      buffer: binding === 0 ? {type: 'uniform', hasDynamicOffset: true, minBindingSize: 32} : {type: [4, 5].includes(binding) ? 'read-only-storage' : 'storage'}}))});
    this.bindings = device.createBindGroup({layout, entries: buffers.map((buffer, binding) => ({binding, resource: binding === 0 ? {buffer, size: 32} : {buffer}}))});
    const pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [layout]});
    const stages = neighbor ? [...STAGES.filter(name => name !== 'gain_adam'), 'mesh_gradient'] : this.model.id === 'residual' ? [...STAGES.filter(name => name !== 'gain_adam'), 'matrix_gradient'] : [...STAGES];
    if (this.model.projection !== undefined) stages.push('project_inputs', 'projection_gradient');
    const pipelines = await Promise.all(stages.map(entryPoint => device.createComputePipelineAsync({label: entryPoint, layout: pipelineLayout, compute: {module, entryPoint}})));
    this.pipelines = Object.fromEntries(stages.map((name, i) => [name, pipelines[i]]));
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);
  }
  settings(view, slot, batch, mode, lr = 0, correction1 = 1, correction2 = 1) {
    const offset = slot * STRIDE;
    for (const [i, value] of [batch.rows, batch.offset ?? 0, batch.bins ?? 0, mode].entries()) view.setUint32(offset + i * 4, value, true);
    for (const [i, value] of [lr, this.decay, correction1, correction2].entries()) view.setFloat32(offset + 16 + i * 4, value, true);
  }
  dispatch(pass, name, x, y = 1, z = 1) { pass.setPipeline(this.pipelines[name]); pass.dispatchWorkgroups(x, y, z); }
  check() { if (this.failure) throw this.failure; }
  async train(samples, {gradientsOnly = false} = {}) {
    const W = this.model.width, residual = Boolean(this.model.blocks);
    this.check();
    if (!samples.length || samples.length > SLOTS) throw new Error('A GPU submission needs 1..32 minibatches.');
    if (gradientsOnly && samples.length !== 1) throw new Error('Gradient verification needs exactly one batch.');
    const uniforms = new ArrayBuffer(SLOTS * STRIDE), view = new DataView(uniforms);
    const packed = new Uint32Array(SLOTS * BATCH_WORDS);
    let cursor = 0;
    const batches = samples.map((sample, slot) => {
      const batch = packBatch(this.data, sample, packed, cursor); cursor = batch.end;
      let lr = 0;
      if (!gradientsOnly) {
        this.step++;
        this.b1 = Math.fround(this.b1 * Math.fround(.9)); this.b2 = Math.fround(this.b2 * Math.fround(.98));
        lr = Math.fround(Math.fround(this.learningRate) * Math.fround(Math.min(this.step / 10, 1)));
      }
      this.settings(view, slot, batch, 1, lr, Math.fround(1 - this.b1), Math.fround(1 - this.b2));
      return batch;
    });
    this.device.queue.writeBuffer(this.uniforms, 0, uniforms, 0, samples.length * STRIDE);
    this.device.queue.writeBuffer(this.order, 0, packed, 0, cursor);
    const encoder = this.device.createCommandEncoder({label: 'Batched optimizer steps'});
    const pass = encoder.beginComputePass();
    for (const [slot, batch] of batches.entries()) {
      pass.setBindGroup(0, this.bindings, [slot * STRIDE]);
      if (this.model.projection !== undefined) this.dispatch(pass, 'project_inputs', 2 * P);
      this.dispatch(pass, 'forward', batch.rows);
      this.dispatch(pass, 'backward', batch.rows);
      this.dispatch(pass, 'readout_gradient', Math.ceil(P / 8), W / 8);
      this.dispatch(pass, 'input_gradient', this.model.rotaryEmbedding ? P : 2 * P);
      if (this.model.projection !== undefined) this.dispatch(pass, 'projection_gradient', Math.ceil(this.model.gain / 128));
      if (this.model.id === 'residual') this.dispatch(pass, 'matrix_gradient', 32, 32, 4);
      if (this.model.meshDepth) this.dispatch(pass, 'mesh_gradient', this.model.blocks * this.model.meshDepth);
      this.dispatch(pass, 'dense_adam', Math.ceil((residual ? this.model.parameters : this.model.gain) / 128));
      if (!residual) this.dispatch(pass, 'gain_adam', 1);
    }
    pass.end(); this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone(); this.check();
  }
  async evaluate() {
    this.check();
    const uniforms = new ArrayBuffer(32); this.settings(new DataView(uniforms), 0, {rows: TOTAL}, 0);
    this.device.queue.writeBuffer(this.uniforms, 0, uniforms);
    const encoder = this.device.createCommandEncoder({label: 'Complete-split evaluation'});
    const pass = encoder.beginComputePass(); pass.setBindGroup(0, this.bindings, [0]);
    if (this.model.projection !== undefined) this.dispatch(pass, 'project_inputs', 2 * P);
    this.dispatch(pass, 'forward', TOTAL); this.dispatch(pass, 'reduce_statistics', 2); pass.end();
    encoder.copyBufferToBuffer(this.statistics, TOTAL * 8, this.metricReadback, 0, 16);
    this.device.queue.submit([encoder.finish()]);
    await this.metricReadback.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(this.metricReadback.getMappedRange()).slice(); this.metricReadback.unmap();
    this.check();
    if (!values.every(Number.isFinite)) throw new Error('Training diverged: metrics are not finite. Reset the experiment.');
    return {step: this.step, epoch: this.step / 10, trainLoss: values[0], trainAccuracy: values[1], heldoutLoss: values[2], heldoutAccuracy: values[3]};
  }
  async readWeights() { return this.read(this.weights, this.weightReadback, this.model.parameters); }
  async read(source, destination, length) {
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(source, 0, destination, 0, length * 4);
    this.device.queue.submit([encoder.finish()]); await destination.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(destination.getMappedRange()).slice(); destination.unmap(); this.check(); return values;
  }
  async readGradients() {
    const PARAMETERS = this.model.parameters;
    const staging = this.buffer('Verification gradient readback', PARAMETERS * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const result = await this.read(this.gradients, staging, PARAMETERS); staging.destroy(); return result;
  }
  async readLogits() {
    const uniforms = new ArrayBuffer(32); this.settings(new DataView(uniforms), 0, {rows: TOTAL}, 2);
    this.device.queue.writeBuffer(this.uniforms, 0, uniforms);
    const encoder = this.device.createCommandEncoder(); const pass = encoder.beginComputePass(); pass.setBindGroup(0, this.bindings, [0]);
    if (this.model.projection !== undefined) this.dispatch(pass, 'project_inputs', 2 * P);
    this.dispatch(pass, 'forward', TOTAL); pass.end(); this.device.queue.submit([encoder.finish()]);
    const staging = this.buffer('Verification logits', TOTAL * P * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const result = await this.read(this.errors, staging, TOTAL * P); staging.destroy(); return result;
  }
  info() { return {description: this.adapter.info.description || this.adapter.info.device || 'WebGPU adapter', backend: 'webgpu', fallback: this.adapter.info.isFallbackAdapter ?? false}; }
  destroy() { for (const buffer of this.buffers) buffer.destroy(); this.device.destroy(); }
}
