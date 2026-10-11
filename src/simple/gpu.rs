use std::sync::Arc;

use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, LaunchContractError, LaunchDimension,
};
use cuda_device::{DisjointSlice, device, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;

use crate::{
    Result,
    gpt::{AdamWOptions, RMSNORM_EPSILON},
    mesh::validate_finite,
};

use super::{Activation, Architecture, Dense, MAX_SIMPLE_ROWS, SimpleConfig};

const IDENTITY: usize = 0;
const RELU: usize = 1;
const SQUARE: usize = 2;
const ADD: usize = 0;
const PRODUCT: usize = 1;
/// Training alternates full and tail minibatches; evaluation uses complete splits.
const WORKSPACE_CAPACITY: usize = 4;

#[cuda_module]
// Flat row-major tensors keep every index explicit.
#[allow(clippy::too_many_arguments)]
// Every kernel is generic, so each launch resolves its own specialization by
// function-item identity and `kernels::load` is safe. Kernels without a
// natural compile-time parameter take an unused `S`, always 0.
mod kernels {
    use super::*;

    #[device]
    fn simple_activate(value: f32, activation: usize) -> f32 {
        if activation == RELU {
            if value > 0.0 { value } else { 0.0 }
        } else if activation == SQUARE {
            value * value
        } else {
            value
        }
    }

    /// The derivative of `simple_activate`, with ReLU's subgradient 0 at the kink.
    #[device]
    fn simple_slope(value: f32, activation: usize) -> f32 {
        if activation == RELU {
            if value > 0.0 { 1.0 } else { 0.0 }
        } else if activation == SQUARE {
            2.0 * value
        } else {
            1.0
        }
    }

    #[device]
    fn simple_warp_output<'a, T>(
        output: &'a mut DisjointSlice<'_, T, thread::WarpIndex>,
    ) -> Option<&'a mut T> {
        if let Some(warp) = thread::warp_index() {
            output.get_mut(warp)
        } else {
            None
        }
    }

    /// Copies table rows `indices[offset..offset + rows]`.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        columns >= 1, rows >= 1, offset + rows <= indices.len(),
        table.len() >= columns, out.len() == rows * columns))]
    pub fn simple_gather<const S: usize>(
        columns: usize,
        rows: usize,
        offset: usize,
        indices: &[u32],
        table: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let row = indices[offset + item / columns] as usize;
            *output = table[row * columns + item % columns];
        }
        let _ = rows;
    }

    /// `out[r][o] = sum_i f(input[r][offset + i]) W[i][o]`; input rows have
    /// `stride` values and `f` is the `ACTIVATION` code.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, inputs >= 1, outputs >= 1, offset + inputs <= stride,
        input.len() == rows * stride, weights.len() == inputs * outputs,
        out.len() == rows * outputs))]
    pub fn simple_dense<const ACTIVATION: usize>(
        rows: usize,
        inputs: usize,
        outputs: usize,
        stride: usize,
        offset: usize,
        input: &[f32],
        weights: &[[f32; 3]],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let start = item / outputs * stride + offset;
            let column = item % outputs;
            let mut total = 0.0;
            for feature in 0..inputs {
                total += simple_activate(input[start + feature], ACTIVATION)
                    * weights[feature * outputs + column][0];
            }
            *output = total;
        }
        let _ = rows;
    }

    /// `out[r][i] = f'(before[r][i]) sum_o gradient[r][o] W[i][o]`.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, inputs >= 1, outputs >= 1,
        gradient.len() == rows * outputs, weights.len() == inputs * outputs,
        before.len() == rows * inputs, out.len() == rows * inputs))]
    pub fn simple_dense_input_gradient<const ACTIVATION: usize>(
        rows: usize,
        inputs: usize,
        outputs: usize,
        gradient: &[f32],
        weights: &[[f32; 3]],
        before: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let start = item / inputs * outputs;
            let row = item % inputs * outputs;
            let mut total = 0.0;
            for column in 0..outputs {
                total += gradient[start + column] * weights[row + column][0];
            }
            *output = total * simple_slope(before[item], ACTIVATION);
        }
        let _ = rows;
    }

    /// `out[i][o] = sum_r f(input[r][offset + i]) gradient[r][o]`. One thread owns
    /// each weight and adds the rows in order, so the result is deterministic.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, inputs >= 1, outputs >= 1, offset + inputs <= stride,
        input.len() == rows * stride, gradient.len() == rows * outputs,
        out.len() == inputs * outputs))]
    pub fn simple_dense_weight_gradient<const ACTIVATION: usize>(
        rows: usize,
        inputs: usize,
        outputs: usize,
        stride: usize,
        offset: usize,
        input: &[f32],
        gradient: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let feature = offset + item / outputs;
            let column = item % outputs;
            let mut total = 0.0;
            for row in 0..rows {
                total += simple_activate(input[row * stride + feature], ACTIVATION)
                    * gradient[row * outputs + column];
            }
            *output = total;
        }
        let _ = inputs;
    }

    /// Elementwise `left + right`, or `left * right` for the `PRODUCT` operation.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        left.len() == out.len(), right.len() == out.len()))]
    pub fn simple_combine<const OPERATION: usize>(
        left: &[f32],
        right: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            *output = if OPERATION == PRODUCT {
                left[item] * right[item]
            } else {
                left[item] + right[item]
            };
        }
    }

    /// One warp per row: the inverse RMS and, backward, `sum_c dn[c] n[c]`.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, width >= 1, input.len() == rows * width,
        gradient.len() == input.len(), normalized.len() == input.len(), out.len() == rows))]
    pub fn simple_rmsnorm_statistics<const BACKWARD: usize>(
        rows: usize,
        width: usize,
        input: &[f32],
        gradient: &[f32],
        normalized: &[f32],
        mut out: DisjointSlice<[f32; 2], thread::WarpIndex>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / 32;
        let lane = index.get() % 32;
        // Rows are whole warps, so every reduction below has all 32 lanes.
        if row >= rows {
            return;
        }
        let start = row * width;
        let mut squares = 0.0;
        let mut dot = 0.0;
        for channel in (lane..width).step_by(32) {
            let value = input[start + channel];
            squares += value * value;
            if BACKWARD == 1 {
                dot += gradient[start + channel] * normalized[start + channel];
            }
        }
        let squares = warp::reduce_sum_f32(squares);
        let dot = warp::reduce_sum_f32(dot);
        if let Some(output) = simple_warp_output(&mut out) {
            *output = [1.0 / (squares / width as f32 + RMSNORM_EPSILON).sqrt(), dot];
        }
    }

    /// Forward `inverse * x`; backward `inverse * (dn - n * dot / width)`.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, width >= 1, input.len() == rows * width,
        normalized.len() == input.len(), statistics.len() == rows,
        out.len() == input.len()))]
    pub fn simple_rmsnorm_apply<const BACKWARD: usize>(
        rows: usize,
        width: usize,
        input: &[f32],
        normalized: &[f32],
        statistics: &[[f32; 2]],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let [inverse, dot] = statistics[item / width];
            *output = inverse
                * if BACKWARD == 1 {
                    input[item] - normalized[item] * dot / width as f32
                } else {
                    input[item]
                };
        }
        let _ = rows;
    }

    /// `(softmax(logits[r]) - onehot(target)) / rows`. Targets are read through
    /// the same minibatch indices as the gathered features.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, classes >= 2, offset + rows <= indices.len(), targets.len() >= 1,
        logits.len() == rows * classes, out.len() == logits.len()))]
    pub fn simple_cross_entropy_gradient<const S: usize>(
        rows: usize,
        classes: usize,
        offset: usize,
        indices: &[u32],
        targets: &[u32],
        logits: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let row = item / classes;
            let start = row * classes;
            let mut maximum = f32::NEG_INFINITY;
            for class in 0..classes {
                if logits[start + class] > maximum {
                    maximum = logits[start + class];
                }
            }
            let mut sum = 0.0;
            for class in 0..classes {
                sum += (logits[start + class] - maximum).exp();
            }
            let target = targets[indices[offset + row] as usize] as usize;
            let indicator = if item - start == target { 1.0 } else { 0.0 };
            *output = ((logits[item] - maximum).exp() / sum - indicator) / rows as f32;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (gradient.len() == state.len()))]
    pub fn simple_adam<const S: usize>(
        learning_rate: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        correction1: f32,
        correction2: f32,
        gradient: &[f32],
        mut state: DisjointSlice<[f32; 3]>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let parameter = index.get();
        if let Some(slot) = state.get_mut(index) {
            let g = gradient[parameter];
            slot[1] = beta1 * slot[1] + (1.0 - beta1) * g;
            slot[2] = beta2 * slot[2] + (1.0 - beta2) * g * g;
            let m = slot[1] / correction1;
            let v = slot[2] / correction2;
            slot[0] -= learning_rate * (m / (v.sqrt() + epsilon) + weight_decay * slot[0]);
        }
    }
}

/// Weights with their AdamW moments, and the latest gradient, for one layer.
struct Parameters {
    state: DeviceBuffer<[f32; 3]>,
    gradient: DeviceBuffer<f32>,
}

/// Encoded equations and answer classes resident on the device.
pub struct DeviceTable {
    features: DeviceBuffer<f32>,
    targets: DeviceBuffer<u32>,
    identity: DeviceBuffer<u32>,
    rows: usize,
}

impl DeviceTable {
    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// Validated row indices into one table, such as every minibatch of an epoch.
pub struct DeviceBatches {
    indices: DeviceBuffer<u32>,
    table_rows: usize,
}

impl DeviceBatches {
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

struct Workspace {
    rows: usize,
    inputs: DeviceBuffer<f32>,
    values: Vec<DeviceBuffer<f32>>,
    adjoints: Vec<DeviceBuffer<f32>>,
    logits: DeviceBuffer<f32>,
    logit_gradient: DeviceBuffer<f32>,
    statistics: DeviceBuffer<[f32; 2]>,
}

impl Workspace {
    fn new(stream: &Arc<CudaStream>, config: SimpleConfig, rows: usize) -> Result<Self> {
        let (values, adjoints) = buffer_widths(config);
        let buffers = |widths: Vec<usize>| -> Result<Vec<DeviceBuffer<f32>>> {
            widths
                .into_iter()
                .map(|width| Ok(DeviceBuffer::zeroed(stream, rows * width)?))
                .collect()
        };
        Ok(Self {
            rows,
            inputs: DeviceBuffer::zeroed(stream, rows * config.input_features())?,
            values: buffers(values)?,
            adjoints: buffers(adjoints)?,
            logits: DeviceBuffer::zeroed(stream, rows * config.modulus)?,
            logit_gradient: DeviceBuffer::zeroed(stream, rows * config.modulus)?,
            statistics: DeviceBuffer::zeroed(stream, rows)?,
        })
    }
}

/// Row widths of the saved forward values and of the backward scratch buffers.
///
/// MLP values are the hidden pre-activations; their gradients alternate
/// between two buffers. A residual MLP saves the streams `h_0..=h_B`, their
/// RMSNorms `n_0..=n_B`, each block's pre-activation `z_b`, and one block
/// output. A bilinear model saves both embeddings and their product.
fn buffer_widths(config: SimpleConfig) -> (Vec<usize>, Vec<usize>) {
    let (width, depth) = (config.width, config.depth);
    match config.architecture {
        Architecture::Mlp => (vec![width; depth], vec![width; 2]),
        Architecture::ResidualMlp => {
            let hidden = width * config.expansion;
            let mut values = vec![width; 2 * depth + 2];
            values.extend(std::iter::repeat_n(hidden, depth));
            values.push(width);
            (values, vec![width, width, hidden, width, width])
        }
        Architecture::Bilinear => (vec![width; 3], vec![width; 3]),
    }
}

struct Kernels {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
}

// Launch wrappers mirror their kernels' argument lists.
#[allow(clippy::too_many_arguments)]
impl Kernels {
    fn gather(
        &self,
        table: &DeviceTable,
        indices: &DeviceBuffer<u32>,
        offset: usize,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let rows = out.len() / (table.features.len() / table.rows);
        let launch = self
            .module
            .prepare_simple_gather::<0>(config::<1>(out.len())?)?;
        self.module.simple_gather(
            &self.stream,
            &launch,
            table.features.len() / table.rows,
            rows,
            offset,
            indices,
            &table.features,
            out,
        )?;
        Ok(())
    }

    /// `columns` holds the input row stride and the first feature the layer
    /// reads; `ACTIVATION` applies to those inputs.
    fn dense<const ACTIVATION: usize>(
        &self,
        layer: Dense,
        rows: usize,
        input: &DeviceBuffer<f32>,
        columns: [usize; 2],
        weights: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [stride, offset] = columns;
        let launch = self
            .module
            .prepare_simple_dense::<ACTIVATION>(config::<1>(out.len())?)?;
        self.module.simple_dense(
            &self.stream,
            &launch,
            rows,
            layer.inputs,
            layer.outputs,
            stride,
            offset,
            input,
            weights,
            out,
        )?;
        Ok(())
    }

    /// `before` holds the pre-activation values of the layer input.
    fn input_gradient<const ACTIVATION: usize>(
        &self,
        layer: Dense,
        rows: usize,
        gradient: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        before: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self
            .module
            .prepare_simple_dense_input_gradient::<ACTIVATION>(config::<1>(out.len())?)?;
        self.module.simple_dense_input_gradient(
            &self.stream,
            &launch,
            rows,
            layer.inputs,
            layer.outputs,
            gradient,
            weights,
            before,
            out,
        )?;
        Ok(())
    }

    fn weight_gradient<const ACTIVATION: usize>(
        &self,
        layer: Dense,
        rows: usize,
        input: &DeviceBuffer<f32>,
        columns: [usize; 2],
        gradient: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [stride, offset] = columns;
        let launch = self
            .module
            .prepare_simple_dense_weight_gradient::<ACTIVATION>(config::<1>(out.len())?)?;
        self.module.simple_dense_weight_gradient(
            &self.stream,
            &launch,
            rows,
            layer.inputs,
            layer.outputs,
            stride,
            offset,
            input,
            gradient,
            out,
        )?;
        Ok(())
    }

    fn combine<const OPERATION: usize>(
        &self,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self
            .module
            .prepare_simple_combine::<OPERATION>(config::<1>(out.len())?)?;
        self.module
            .simple_combine(&self.stream, &launch, left, right, out)?;
        Ok(())
    }

    /// Forward reads `[x, x, x]`; backward reads `[x, dn, n]` and writes `dx`.
    fn rmsnorm<const BACKWARD: usize>(
        &self,
        rows: usize,
        width: usize,
        inputs: [&DeviceBuffer<f32>; 3],
        statistics: &mut DeviceBuffer<[f32; 2]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [input, gradient, normalized] = inputs;
        let launch = self
            .module
            .prepare_simple_rmsnorm_statistics::<BACKWARD>(config::<32>(rows)?)?;
        self.module.simple_rmsnorm_statistics(
            &self.stream,
            &launch,
            rows,
            width,
            input,
            gradient,
            normalized,
            statistics,
        )?;
        let launch = self
            .module
            .prepare_simple_rmsnorm_apply::<BACKWARD>(config::<1>(out.len())?)?;
        self.module.simple_rmsnorm_apply(
            &self.stream,
            &launch,
            rows,
            width,
            if BACKWARD == 1 { gradient } else { input },
            normalized,
            statistics,
            out,
        )?;
        Ok(())
    }

    fn cross_entropy_gradient(
        &self,
        classes: usize,
        offset: usize,
        indices: &DeviceBuffer<u32>,
        targets: &DeviceBuffer<u32>,
        logits: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self
            .module
            .prepare_simple_cross_entropy_gradient::<0>(config::<1>(out.len())?)?;
        self.module.simple_cross_entropy_gradient(
            &self.stream,
            &launch,
            logits.len() / classes,
            classes,
            offset,
            indices,
            targets,
            logits,
            out,
        )?;
        Ok(())
    }

    fn adam(
        &self,
        optimizer: AdamWOptions,
        corrections: [f32; 2],
        group: &mut Parameters,
    ) -> Result<()> {
        let launch = self
            .module
            .prepare_simple_adam::<0>(config::<1>(group.state.len())?)?;
        self.module.simple_adam(
            &self.stream,
            &launch,
            optimizer.learning_rate,
            optimizer.weight_decay,
            optimizer.beta1,
            optimizer.beta2,
            optimizer.epsilon,
            corrections[0],
            corrections[1],
            &group.gradient,
            &mut group.state,
        )?;
        Ok(())
    }
}

/// An attention-free model trained entirely on the device. Parameters,
/// AdamW moments, encoded splits, and epoch indices stay resident; a training
/// step copies nothing to or from the host.
pub struct GpuSimpleModel {
    config: SimpleConfig,
    layers: Vec<Dense>,
    kernels: Kernels,
    parameters: Vec<Parameters>,
    workspaces: Vec<Workspace>,
    beta1_power: f32,
    beta2_power: f32,
}

impl GpuSimpleModel {
    pub fn new(device: usize, config: SimpleConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        let context = CudaContext::new(device)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let layers = config.layers();
        let initial = config.initial_parameters(seed);
        let mut parameters = Vec::with_capacity(layers.len());
        let mut offset = 0;
        for layer in &layers {
            let state: Vec<_> = initial[offset..offset + layer.parameters()]
                .iter()
                .map(|&p| [p, 0.0, 0.0])
                .collect();
            parameters.push(Parameters {
                state: DeviceBuffer::from_host(&stream, &state)?,
                gradient: DeviceBuffer::zeroed(&stream, layer.parameters())?,
            });
            offset += layer.parameters();
        }
        Ok(Self {
            config,
            layers,
            kernels: Kernels { stream, module },
            parameters,
            workspaces: Vec::new(),
            beta1_power: 1.0,
            beta2_power: 1.0,
        })
    }

    pub fn config(&self) -> SimpleConfig {
        self.config
    }

    pub fn parameters(&self) -> Result<Vec<f32>> {
        let mut values = Vec::with_capacity(self.config.parameter_count());
        for group in &self.parameters {
            values.extend(
                group
                    .state
                    .to_host_vec(&self.kernels.stream)?
                    .iter()
                    .map(|s| s[0]),
            );
        }
        validate_finite(&values, "parameters")?;
        Ok(values)
    }

    /// Sets weights and resets AdamW's moments and step count.
    pub fn set_parameters(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.config.parameter_count() {
            return Err("parameter count does not match the model".into());
        }
        validate_finite(values, "parameters")?;
        let mut offset = 0;
        for (layer, group) in self.layers.iter().zip(&mut self.parameters) {
            let state: Vec<_> = values[offset..offset + layer.parameters()]
                .iter()
                .map(|&p| [p, 0.0, 0.0])
                .collect();
            group.state.copy_from_host(&self.kernels.stream, &state)?;
            offset += layer.parameters();
        }
        self.beta1_power = 1.0;
        self.beta2_power = 1.0;
        Ok(())
    }

    /// Uploads encoded rows from `SimpleConfig::encode` and their answer classes.
    pub fn upload_table(&self, features: &[f32], targets: &[u32]) -> Result<DeviceTable> {
        let rows = targets.len();
        if rows == 0 || rows > MAX_SIMPLE_ROWS {
            return Err(format!("tables need 1..={MAX_SIMPLE_ROWS} rows").into());
        }
        if features.len() != rows * self.config.input_features() {
            return Err("each table row needs the model's input features".into());
        }
        validate_finite(features, "features")?;
        if targets
            .iter()
            .any(|&target| target as usize >= self.config.modulus)
        {
            return Err("answer classes must be below the modulus".into());
        }
        let stream = &self.kernels.stream;
        let identity: Vec<_> = (0..rows as u32).collect();
        Ok(DeviceTable {
            features: DeviceBuffer::from_host(stream, features)?,
            targets: DeviceBuffer::from_host(stream, targets)?,
            identity: DeviceBuffer::from_host(stream, &identity)?,
            rows,
        })
    }

    /// Uploads row indices for later minibatches, checking each against `table`.
    pub fn upload_batches(&self, table: &DeviceTable, indices: &[u32]) -> Result<DeviceBatches> {
        if indices.is_empty() || indices.len() > MAX_SIMPLE_ROWS {
            return Err(format!("batches need 1..={MAX_SIMPLE_ROWS} indices").into());
        }
        if indices.iter().any(|&index| index as usize >= table.rows) {
            return Err("batch index is outside the table".into());
        }
        Ok(DeviceBatches {
            indices: DeviceBuffer::from_host(&self.kernels.stream, indices)?,
            table_rows: table.rows,
        })
    }

    /// Logits for every table row in order, `modulus` classes per row.
    pub fn logits(&mut self, table: &DeviceTable) -> Result<Vec<f32>> {
        let mut workspace = self.take_workspace(table.rows)?;
        let result = (|| {
            self.kernels
                .gather(table, &table.identity, 0, &mut workspace.inputs)?;
            self.forward(&mut workspace)?;
            let logits = workspace.logits.to_host_vec(&self.kernels.stream)?;
            validate_finite(&logits, "logits")?;
            Ok(logits)
        })();
        self.workspaces.push(workspace);
        result
    }

    /// The mean answer cross-entropy and every parameter gradient on the
    /// selected rows. Diagnostic only: training never copies gradients.
    pub fn loss_and_gradient(
        &mut self,
        table: &DeviceTable,
        indices: &[u32],
    ) -> Result<(f64, Vec<f64>)> {
        let batches = self.upload_batches(table, indices)?;
        let logits = self.gradients(table, &batches, 0, indices.len(), true)?;
        validate_finite(&logits, "logits")?;
        let targets = table.targets.to_host_vec(&self.kernels.stream)?;
        let mut loss = 0.0;
        for (row, &index) in logits.chunks_exact(self.config.modulus).zip(indices) {
            let maximum = row
                .iter()
                .fold(f64::NEG_INFINITY, |a, &x| a.max(f64::from(x)));
            let sum = row
                .iter()
                .map(|&x| (f64::from(x) - maximum).exp())
                .sum::<f64>();
            loss += sum.ln() + maximum - f64::from(row[targets[index as usize] as usize]);
        }
        let mut gradient = Vec::with_capacity(self.config.parameter_count());
        for group in &self.parameters {
            let values = group.gradient.to_host_vec(&self.kernels.stream)?;
            validate_finite(&values, "gradients")?;
            gradient.extend(values.into_iter().map(f64::from));
        }
        Ok((loss / indices.len() as f64, gradient))
    }

    /// One AdamW step on rows `batches[offset..offset + rows]` of `table`.
    /// Repeated calls continue the same optimizer state.
    pub fn train_batch(
        &mut self,
        table: &DeviceTable,
        batches: &DeviceBatches,
        offset: usize,
        rows: usize,
        optimizer: AdamWOptions,
    ) -> Result<()> {
        optimizer.validate()?;
        self.gradients(table, batches, offset, rows, false)?;
        self.beta1_power *= optimizer.beta1;
        self.beta2_power *= optimizer.beta2;
        let corrections = [1.0 - self.beta1_power, 1.0 - self.beta2_power];
        for group in &mut self.parameters {
            self.kernels.adam(optimizer, corrections, group)?;
        }
        Ok(())
    }

    fn take_workspace(&mut self, rows: usize) -> Result<Workspace> {
        if let Some(index) = self.workspaces.iter().position(|w| w.rows == rows) {
            return Ok(self.workspaces.remove(index));
        }
        if self.workspaces.len() == WORKSPACE_CAPACITY {
            self.workspaces.remove(0);
        }
        Workspace::new(&self.kernels.stream, self.config, rows)
    }

    /// Computes every layer's gradient; returns the logits only when requested.
    fn gradients(
        &mut self,
        table: &DeviceTable,
        batches: &DeviceBatches,
        offset: usize,
        rows: usize,
        read_logits: bool,
    ) -> Result<Vec<f32>> {
        if batches.table_rows != table.rows {
            return Err("batch indices belong to a table with another row count".into());
        }
        if rows == 0
            || rows > MAX_SIMPLE_ROWS
            || offset
                .checked_add(rows)
                .is_none_or(|end| end > batches.len())
        {
            return Err("minibatch is outside the uploaded indices".into());
        }
        let mut workspace = self.take_workspace(rows)?;
        let result = (|| {
            self.kernels
                .gather(table, &batches.indices, offset, &mut workspace.inputs)?;
            self.forward(&mut workspace)?;
            self.kernels.cross_entropy_gradient(
                self.config.modulus,
                offset,
                &batches.indices,
                &table.targets,
                &workspace.logits,
                &mut workspace.logit_gradient,
            )?;
            self.backward(&mut workspace)?;
            if read_logits {
                Ok(workspace.logits.to_host_vec(&self.kernels.stream)?)
            } else {
                Ok(Vec::new())
            }
        })();
        self.workspaces.push(workspace);
        result
    }

    fn forward(&self, workspace: &mut Workspace) -> Result<()> {
        match self.config.activation {
            Activation::Relu => self.forward_with::<RELU>(workspace),
            Activation::Square => self.forward_with::<SQUARE>(workspace),
        }
    }

    fn backward(&mut self, workspace: &mut Workspace) -> Result<()> {
        match self.config.activation {
            Activation::Relu => self.backward_with::<RELU>(workspace),
            Activation::Square => self.backward_with::<SQUARE>(workspace),
        }
    }

    fn forward_with<const ACTIVATION: usize>(&self, workspace: &mut Workspace) -> Result<()> {
        let c = self.config;
        let rows = workspace.rows;
        let (k, layers) = (&self.kernels, &self.layers);
        let state = |layer: usize| &self.parameters[layer].state;
        let input = [c.input_features(), 0];
        let values = &mut workspace.values;
        match c.architecture {
            Architecture::Mlp => {
                let depth = c.depth;
                k.dense::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    input,
                    state(0),
                    &mut values[0],
                )?;
                for (layer, &shape) in layers.iter().enumerate().take(depth).skip(1) {
                    let [before, after] = values.get_disjoint_mut([layer - 1, layer])?;
                    k.dense::<ACTIVATION>(shape, rows, before, [c.width, 0], state(layer), after)?;
                }
                k.dense::<ACTIVATION>(
                    layers[depth],
                    rows,
                    &values[depth - 1],
                    [c.width, 0],
                    state(depth),
                    &mut workspace.logits,
                )?;
            }
            Architecture::ResidualMlp => {
                let blocks = c.depth;
                let hidden = c.width * c.expansion;
                // Value indices: h_k = k, n_k = blocks + 1 + k, z_b = 2 blocks + 2 + b.
                let (normalized, expanded) = (blocks + 1, 2 * blocks + 2);
                let branch = 3 * blocks + 2;
                k.dense::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    input,
                    state(0),
                    &mut values[0],
                )?;
                for block in 0..blocks {
                    let (up, down) = (1 + 2 * block, 2 + 2 * block);
                    let [h, n] = values.get_disjoint_mut([block, normalized + block])?;
                    k.rmsnorm::<0>(rows, c.width, [&*h, &*h, &*h], &mut workspace.statistics, n)?;
                    let [n, z] = values.get_disjoint_mut([normalized + block, expanded + block])?;
                    k.dense::<IDENTITY>(layers[up], rows, n, [c.width, 0], state(up), z)?;
                    let [z, y] = values.get_disjoint_mut([expanded + block, branch])?;
                    k.dense::<ACTIVATION>(layers[down], rows, z, [hidden, 0], state(down), y)?;
                    let [h, y, next] = values.get_disjoint_mut([block, branch, block + 1])?;
                    k.combine::<ADD>(h, y, next)?;
                }
                let [h, n] = values.get_disjoint_mut([blocks, normalized + blocks])?;
                k.rmsnorm::<0>(rows, c.width, [&*h, &*h, &*h], &mut workspace.statistics, n)?;
                let readout = 1 + 2 * blocks;
                k.dense::<IDENTITY>(
                    layers[readout],
                    rows,
                    &values[normalized + blocks],
                    [c.width, 0],
                    state(readout),
                    &mut workspace.logits,
                )?;
            }
            Architecture::Bilinear => {
                let features = c.operand_features();
                let [left, right, product] = values.get_disjoint_mut([0, 1, 2])?;
                k.dense::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    [2 * features, 0],
                    state(0),
                    left,
                )?;
                k.dense::<IDENTITY>(
                    layers[1],
                    rows,
                    &workspace.inputs,
                    [2 * features, features],
                    state(1),
                    right,
                )?;
                k.combine::<PRODUCT>(left, right, product)?;
                k.dense::<IDENTITY>(
                    layers[2],
                    rows,
                    product,
                    [c.width, 0],
                    state(2),
                    &mut workspace.logits,
                )?;
            }
        }
        Ok(())
    }

    fn backward_with<const ACTIVATION: usize>(&mut self, workspace: &mut Workspace) -> Result<()> {
        let c = self.config;
        let rows = workspace.rows;
        let (k, layers, parameters) = (&self.kernels, &self.layers, &mut self.parameters);
        let input = [c.input_features(), 0];
        let values = &workspace.values;
        let adjoints = &mut workspace.adjoints;
        let logits = &workspace.logit_gradient;
        match c.architecture {
            Architecture::Mlp => {
                // The gradient of hidden pre-activation z_k lives in adjoints[k % 2].
                let depth = c.depth;
                k.weight_gradient::<ACTIVATION>(
                    layers[depth],
                    rows,
                    &values[depth - 1],
                    [c.width, 0],
                    logits,
                    &mut parameters[depth].gradient,
                )?;
                k.input_gradient::<ACTIVATION>(
                    layers[depth],
                    rows,
                    logits,
                    &parameters[depth].state,
                    &values[depth - 1],
                    &mut adjoints[(depth - 1) % 2],
                )?;
                for layer in (1..depth).rev() {
                    let [upstream, target] =
                        adjoints.get_disjoint_mut([layer % 2, (layer - 1) % 2])?;
                    k.weight_gradient::<ACTIVATION>(
                        layers[layer],
                        rows,
                        &values[layer - 1],
                        [c.width, 0],
                        upstream,
                        &mut parameters[layer].gradient,
                    )?;
                    k.input_gradient::<ACTIVATION>(
                        layers[layer],
                        rows,
                        upstream,
                        &parameters[layer].state,
                        &values[layer - 1],
                        target,
                    )?;
                }
                k.weight_gradient::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    input,
                    &adjoints[0],
                    &mut parameters[0].gradient,
                )?;
            }
            Architecture::ResidualMlp => {
                const DN: usize = 0;
                const DH: usize = 1;
                const DZ: usize = 2;
                const BRANCH: usize = 3;
                const SUM: usize = 4;
                let blocks = c.depth;
                let hidden = c.width * c.expansion;
                let (normalized, expanded) = (blocks + 1, 2 * blocks + 2);
                let readout = 1 + 2 * blocks;
                let last = &values[normalized + blocks];
                k.weight_gradient::<IDENTITY>(
                    layers[readout],
                    rows,
                    last,
                    [c.width, 0],
                    logits,
                    &mut parameters[readout].gradient,
                )?;
                k.input_gradient::<IDENTITY>(
                    layers[readout],
                    rows,
                    logits,
                    &parameters[readout].state,
                    last,
                    &mut adjoints[DN],
                )?;
                let [dn, dh] = adjoints.get_disjoint_mut([DN, DH])?;
                k.rmsnorm::<1>(
                    rows,
                    c.width,
                    [&values[blocks], &*dn, last],
                    &mut workspace.statistics,
                    dh,
                )?;
                for block in (0..blocks).rev() {
                    let (up, down) = (1 + 2 * block, 2 + 2 * block);
                    let (n, z) = (&values[normalized + block], &values[expanded + block]);
                    let [dh, dz] = adjoints.get_disjoint_mut([DH, DZ])?;
                    k.weight_gradient::<ACTIVATION>(
                        layers[down],
                        rows,
                        z,
                        [hidden, 0],
                        dh,
                        &mut parameters[down].gradient,
                    )?;
                    k.input_gradient::<ACTIVATION>(
                        layers[down],
                        rows,
                        dh,
                        &parameters[down].state,
                        z,
                        dz,
                    )?;
                    let [dz, dn] = adjoints.get_disjoint_mut([DZ, DN])?;
                    k.weight_gradient::<IDENTITY>(
                        layers[up],
                        rows,
                        n,
                        [c.width, 0],
                        dz,
                        &mut parameters[up].gradient,
                    )?;
                    k.input_gradient::<IDENTITY>(
                        layers[up],
                        rows,
                        dz,
                        &parameters[up].state,
                        n,
                        dn,
                    )?;
                    let [dn, branch] = adjoints.get_disjoint_mut([DN, BRANCH])?;
                    k.rmsnorm::<1>(
                        rows,
                        c.width,
                        [&values[block], &*dn, n],
                        &mut workspace.statistics,
                        branch,
                    )?;
                    // The residual stream receives both the skip and the branch gradient.
                    let [dh, branch, sum] = adjoints.get_disjoint_mut([DH, BRANCH, SUM])?;
                    k.combine::<ADD>(dh, branch, sum)?;
                    adjoints.swap(DH, SUM);
                }
                k.weight_gradient::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    input,
                    &adjoints[DH],
                    &mut parameters[0].gradient,
                )?;
            }
            Architecture::Bilinear => {
                let features = c.operand_features();
                let product = &values[2];
                k.weight_gradient::<IDENTITY>(
                    layers[2],
                    rows,
                    product,
                    [c.width, 0],
                    logits,
                    &mut parameters[2].gradient,
                )?;
                k.input_gradient::<IDENTITY>(
                    layers[2],
                    rows,
                    logits,
                    &parameters[2].state,
                    product,
                    &mut adjoints[0],
                )?;
                let [dp, dl, dr] = adjoints.get_disjoint_mut([0, 1, 2])?;
                k.combine::<PRODUCT>(dp, &values[1], dl)?;
                k.combine::<PRODUCT>(dp, &values[0], dr)?;
                k.weight_gradient::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    [2 * features, 0],
                    dl,
                    &mut parameters[0].gradient,
                )?;
                k.weight_gradient::<IDENTITY>(
                    layers[1],
                    rows,
                    &workspace.inputs,
                    [2 * features, features],
                    dr,
                    &mut parameters[1].gradient,
                )?;
            }
        }
        Ok(())
    }
}

fn config<const LANES: usize>(
    items: usize,
) -> std::result::Result<LaunchConfig1D, LaunchContractError> {
    const {
        assert!(LANES == 1 || LANES == 32);
    }
    // Divide before converting: legal tensors can exceed u32::MAX elements.
    let blocks = items.div_ceil(128 / LANES).try_into().map_err(|_| {
        LaunchContractError::DimensionProductOverflow {
            kernel: "simple model launch",
            dimension: LaunchDimension::Grid,
        }
    })?;
    Ok(LaunchConfig1D::new(blocks, 128, 0))
}

mod verification {
    use super::*;
    use crate::{
        gpt::{EquationDataset, ModularOperation},
        simple::{Activation, Encoding, ReferenceSimpleModel},
    };

    impl GpuSimpleModel {
        /// Checks every kernel path against f64 automatic differentiation:
        /// logits, the loss, all layer gradients, gathered and offset
        /// minibatches with repeated rows, AdamW steps, bitwise repeatability,
        /// input validation, and launch contracts.
        pub fn verify(device: usize) -> Result<()> {
            for config in configurations() {
                let [logits, gradients] = verify_model(device, config)?;
                println!(
                    "PASS {} {} {} width={} depth={} expansion={} modulus={}: logits (max error {logits:.1e} of max), loss, gradients (max error {gradients:.1e} of max) for 1/37/130 gathered rows, offset AdamW steps, repeatability",
                    config.architecture,
                    config.activation,
                    config.encoding,
                    config.width,
                    config.depth,
                    config.expansion,
                    config.modulus
                );
            }
            verify_validation(device)?;
            println!(
                "PASS simple-model validation: tables, indices, minibatch ranges, parameters, launch contracts"
            );
            Ok(())
        }
    }

    fn configurations() -> Vec<SimpleConfig> {
        let base = SimpleConfig {
            modulus: 7,
            width: 5,
            depth: 1,
            ..SimpleConfig::default()
        };
        let residual = SimpleConfig {
            architecture: Architecture::ResidualMlp,
            ..base
        };
        let bilinear = SimpleConfig {
            architecture: Architecture::Bilinear,
            ..base
        };
        vec![
            base,
            SimpleConfig {
                activation: Activation::Square,
                encoding: Encoding::Binary,
                modulus: 11,
                width: 33,
                depth: 2,
                ..base
            },
            SimpleConfig {
                encoding: Encoding::Scalar,
                width: 4,
                depth: 3,
                ..base
            },
            SimpleConfig {
                activation: Activation::Square,
                encoding: Encoding::DiscreteLog,
                modulus: 13,
                width: 40,
                ..base
            },
            // The experiments' shapes: 194 one-hot inputs and 97 classes.
            SimpleConfig {
                modulus: 97,
                width: 12,
                depth: 2,
                ..base
            },
            SimpleConfig {
                width: 6,
                expansion: 4,
                ..residual
            },
            // Widths above one warp exercise strided RMSNorm reductions.
            SimpleConfig {
                activation: Activation::Square,
                encoding: Encoding::DiscreteLog,
                modulus: 11,
                width: 70,
                expansion: 1,
                ..residual
            },
            SimpleConfig {
                modulus: 97,
                width: 8,
                depth: 2,
                expansion: 4,
                ..residual
            },
            SimpleConfig {
                width: 3,
                ..bilinear
            },
            SimpleConfig {
                encoding: Encoding::Binary,
                modulus: 97,
                width: 17,
                ..bilinear
            },
            SimpleConfig {
                encoding: Encoding::Scalar,
                modulus: 5,
                width: 2,
                ..bilinear
            },
        ]
    }

    /// The largest absolute difference as a fraction of the largest expected value.
    fn relative_error(name: &str, expected: &[f64], actual: &[f64], tolerance: f64) -> Result<f64> {
        if expected.len() != actual.len() || actual.iter().any(|x| !x.is_finite()) {
            return Err(format!("{name}: missing or nonfinite values").into());
        }
        let scale = expected.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        if scale == 0.0 {
            return Err(format!("{name}: the reference is identically zero").into());
        }
        let mut worst = 0.0_f64;
        for (index, (&e, &a)) in expected.iter().zip(actual).enumerate() {
            let error = (e - a).abs() / scale;
            if error > tolerance {
                return Err(format!(
                    "{name} {index}: expected {e:.9e}, got {a:.9e} ({error:.1e} of max)"
                )
                .into());
            }
            worst = worst.max(error);
        }
        Ok(worst)
    }

    fn verify_model(device: usize, config: SimpleConfig) -> Result<[f64; 2]> {
        let mut model = GpuSimpleModel::new(device, config, 11)?;
        let initial = ReferenceSimpleModel::new(config, 11)?;
        if model.parameters()? != initial.parameters() {
            return Err("GPU initialization differs from the reference".into());
        }
        // Larger weights make every nonlinearity matter.
        let parameters: Vec<_> = initial.parameters().iter().map(|&p| 2.0 * p).collect();
        model.set_parameters(&parameters)?;
        let reference = ReferenceSimpleModel::from_parameters(config, parameters.clone())?;
        let dataset = EquationDataset::new(config.modulus, ModularOperation::Divide, 0.5, 7)?;
        let equations = dataset.split(false);
        let (features, targets) = config.encode(&equations[..equations.len().min(64)])?;
        let table = model.upload_table(&features, &targets)?;
        let columns = config.input_features();

        let logits = model.logits(&table)?;
        if logits != model.logits(&table)? {
            return Err("logits are not bitwise repeatable".into());
        }
        let expected: Vec<_> = reference.logits(&features)?.into_iter().flatten().collect();
        let actual: Vec<_> = logits.iter().map(|&x| f64::from(x)).collect();
        let logit_error = relative_error("logit", &expected, &actual, 1e-5)?;

        let mut gradient_error = 0.0_f64;
        for rows in [1, 37, 130] {
            // Repeated rows in an arbitrary order exercise the gather.
            let indices: Vec<_> = (0..rows)
                .map(|i| ((i * 7 + 3) % table.rows()) as u32)
                .collect();
            let mut gathered = Vec::with_capacity(rows * columns);
            for &index in &indices {
                let start = index as usize * columns;
                gathered.extend_from_slice(&features[start..start + columns]);
            }
            let answers: Vec<_> = indices.iter().map(|&i| targets[i as usize]).collect();
            let (cpu_loss, cpu_gradient) = reference.loss_and_gradient(&gathered, &answers)?;
            let (gpu_loss, gpu_gradient) = model.loss_and_gradient(&table, &indices)?;
            if (cpu_loss - gpu_loss).abs() > 1e-5 * (1.0 + cpu_loss.abs()) {
                return Err(
                    format!("loss rows={rows}: expected {cpu_loss}, got {gpu_loss}").into(),
                );
            }
            gradient_error = gradient_error.max(relative_error(
                &format!("gradient rows={rows} parameter"),
                &cpu_gradient,
                &gpu_gradient,
                2e-4,
            )?);
        }

        // The minibatch sits at an offset inside a longer index list, as in an epoch.
        let indices: Vec<_> = (0..37)
            .map(|i| ((i * 5 + 1) % table.rows()) as u32)
            .collect();
        let mut epoch: Vec<_> = (0..5).map(|i| (i % table.rows()) as u32).collect();
        epoch.extend_from_slice(&indices);
        epoch.extend_from_slice(&[0, 1 % table.rows() as u32]);
        let batches = model.upload_batches(&table, &epoch)?;
        let optimizer = AdamWOptions {
            learning_rate: 0.01,
            weight_decay: 0.1,
            beta2: 0.98,
            ..AdamWOptions::default()
        };
        model.set_parameters(&parameters)?;
        let mut moments = vec![[0.0_f32; 2]; parameters.len()];
        let (mut b1, mut b2) = (1.0_f32, 1.0_f32);
        let mut first_step = Vec::new();
        for step in 1..=3 {
            let mut expected = model.parameters()?;
            let (_, gradient) = model.loss_and_gradient(&table, &indices)?;
            // Check the optimizer with its actual FP32 inputs.
            b1 *= optimizer.beta1;
            b2 *= optimizer.beta2;
            for ((parameter, moment), gradient) in
                expected.iter_mut().zip(&mut moments).zip(gradient)
            {
                let gradient = gradient as f32;
                moment[0] = optimizer.beta1 * moment[0] + (1.0 - optimizer.beta1) * gradient;
                moment[1] =
                    optimizer.beta2 * moment[1] + (1.0 - optimizer.beta2) * gradient * gradient;
                *parameter -= optimizer.learning_rate
                    * (moment[0]
                        / (1.0 - b1)
                        / ((moment[1] / (1.0 - b2)).sqrt() + optimizer.epsilon)
                        + optimizer.weight_decay * *parameter);
            }
            model.train_batch(&table, &batches, 5, indices.len(), optimizer)?;
            let actual = model.parameters()?;
            for (index, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
                if (e - a).abs() > 1e-6 * (1.0 + e.abs()) {
                    return Err(format!(
                        "AdamW step={step} parameter={index}: expected {e}, got {a}"
                    )
                    .into());
                }
            }
            if step == 1 {
                first_step = actual;
            }
        }
        // Resetting the parameters also resets AdamW, and the step repeats bitwise.
        model.set_parameters(&parameters)?;
        model.train_batch(&table, &batches, 5, indices.len(), optimizer)?;
        if model.parameters()? != first_step {
            return Err("AdamW steps are not bitwise repeatable after a reset".into());
        }
        Ok([logit_error, gradient_error])
    }

    fn verify_validation(device: usize) -> Result<()> {
        let config = SimpleConfig {
            modulus: 7,
            width: 4,
            ..SimpleConfig::default()
        };
        for invalid in [
            SimpleConfig { width: 0, ..config },
            SimpleConfig {
                modulus: 1,
                ..config
            },
        ] {
            if GpuSimpleModel::new(device, invalid, 1).is_ok() {
                return Err("model accepted an invalid configuration".into());
            }
        }
        let mut model = GpuSimpleModel::new(device, config, 1)?;
        let columns = config.input_features();
        let features = vec![0.5; 3 * columns];
        let table = model.upload_table(&features, &[0, 6, 3])?;
        let mut nan = features.clone();
        nan[1] = f32::NAN;
        let parameters = model.parameters()?;
        let mut nonfinite = parameters.clone();
        nonfinite[0] = f32::INFINITY;
        let short = model.upload_table(&features[..columns], &[1])?;
        let foreign = model.upload_batches(&short, &[0])?;
        let batches = model.upload_batches(&table, &[2, 0, 1])?;
        let optimizer = AdamWOptions::default();
        if model.upload_table(&[], &[]).is_ok()
            || model.upload_table(&features[1..], &[0, 6, 3]).is_ok()
            || model.upload_table(&nan, &[0, 6, 3]).is_ok()
            || model.upload_table(&features, &[0, 7, 3]).is_ok()
            || model.upload_batches(&table, &[]).is_ok()
            || model.upload_batches(&table, &[3]).is_ok()
            || model.train_batch(&table, &batches, 2, 2, optimizer).is_ok()
            || model.train_batch(&table, &batches, 0, 0, optimizer).is_ok()
            || model
                .train_batch(&table, &batches, usize::MAX, 1, optimizer)
                .is_ok()
            || model.train_batch(&table, &foreign, 0, 1, optimizer).is_ok()
            || model
                .train_batch(
                    &table,
                    &batches,
                    0,
                    3,
                    AdamWOptions {
                        beta1: 1.0,
                        ..optimizer
                    },
                )
                .is_ok()
            || model.loss_and_gradient(&table, &[3]).is_ok()
            || model.set_parameters(&parameters[1..]).is_ok()
            || model.set_parameters(&nonfinite).is_ok()
        {
            return Err("simple-model validation accepted invalid input".into());
        }
        if model.parameters()? != parameters {
            return Err("rejected input changed the parameters".into());
        }
        // Prepared launches check buffer sizes before the kernels run. Each
        // call below breaks exactly one requirement.
        let k = &model.kernels;
        let layer = model.layers[0];
        let input = DeviceBuffer::<f32>::zeroed(&k.stream, 2 * columns)?;
        let state = &model.parameters[0].state;
        let mut exact = DeviceBuffer::<f32>::zeroed(&k.stream, 2 * layer.outputs)?;
        let mut long = DeviceBuffer::<f32>::zeroed(&k.stream, 2 * layer.outputs + 1)?;
        let mut gradient = DeviceBuffer::<f32>::zeroed(&k.stream, layer.parameters())?;
        let mut statistics = DeviceBuffer::<[f32; 2]>::zeroed(&k.stream, 1)?;
        let mut normalized = DeviceBuffer::<f32>::zeroed(&k.stream, 2 * columns)?;
        k.dense::<IDENTITY>(layer, 2, &input, [columns, 0], state, &mut exact)?;
        k.weight_gradient::<IDENTITY>(layer, 2, &input, [columns, 0], &exact, &mut gradient)?;
        k.gather(&table, &batches.indices, 1, &mut normalized)?;
        if k.dense::<IDENTITY>(layer, 2, &input, [columns, 0], state, &mut long)
            .is_ok()
            || k.dense::<IDENTITY>(layer, 2, &input, [columns, 1], state, &mut exact)
                .is_ok()
            || k.weight_gradient::<IDENTITY>(layer, 2, &input, [columns, 0], &long, &mut gradient)
                .is_ok()
            || k.gather(&table, &batches.indices, 2, &mut normalized)
                .is_ok()
            || k.rmsnorm::<0>(
                2,
                columns,
                [&input, &input, &input],
                &mut statistics,
                &mut normalized,
            )
            .is_ok()
        {
            return Err("a launch contract accepted mismatched buffers".into());
        }
        Ok(())
    }
}
