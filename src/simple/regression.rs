//! Scalar-input, scalar-output ReLU MLPs. Training uses CUDA throughout;
//! the independent f64 tape is used only by verification.
use super::{
    ADD, DeviceBatches, IDENTITY, Kernels, Launches, Parameters, config, dense_tiled, kernels,
};
use crate::{
    Result,
    gpt::{
        AdamWOptions,
        reference::{Tape, rmsnorm},
    },
    mesh::{Random, validate_finite},
    simple::{Dense, MAX_SIMPLE_ROWS},
};
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegressionConfig {
    pub width: usize,
    /// Total ReLU hidden layers, including the two-feature input projection.
    pub depth: usize,
    /// Normalize hidden inputs to width-preserving projections, never the raw pair.
    pub pre_rmsnorm: bool,
    /// Skip only width-preserving hidden layers; the input and readout have no skip.
    pub residual: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegressionBiasInitialization {
    Zero,
    Uniform,
    Domain,
}

impl RegressionBiasInitialization {
    pub fn name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Uniform => "uniform",
            Self::Domain => "domain",
        }
    }
}

impl RegressionConfig {
    pub fn validate(self) -> Result<()> {
        if !(1..=1024).contains(&self.width) || !(1..=16).contains(&self.depth) {
            return Err("regression width must be 1..1024 and hidden depth 1..16".into());
        }
        if self.depth == 1 && (self.pre_rmsnorm || self.residual) {
            return Err(
                "one hidden layer has no width-preserving layer to normalize or skip".into(),
            );
        }
        Ok(())
    }

    fn validate_parameters(self, parameters: &[f32]) -> Result<()> {
        self.validate()?;
        if parameters.len() != self.parameter_count() {
            return Err("regression parameter count does not match the architecture".into());
        }
        validate_finite(parameters, "regression parameters")
    }

    pub fn layers(self) -> Vec<Dense> {
        let mut layers = vec![Dense {
            inputs: 2,
            outputs: self.width,
        }];
        layers.extend((1..self.depth).map(|_| Dense {
            inputs: self.width,
            outputs: self.width,
        }));
        layers.push(Dense {
            inputs: self.width,
            outputs: 1,
        });
        layers
    }

    pub fn parameter_count(self) -> usize {
        self.layers()
            .iter()
            .map(|layer| layer.parameters() + layer.outputs)
            .sum()
    }

    /// Independent Kaiming normal hidden weights, linear-gain output weights,
    /// zero biases. Flags never change the random stream or initial weights.
    pub fn initial_parameters(self, seed: u64) -> Vec<f32> {
        let mut random = Random::new(seed ^ 0x7265677265730001);
        let mut parameters = Vec::with_capacity(self.parameter_count());
        for (index, layer) in self.layers().iter().enumerate() {
            let gain = if index == self.depth {
                1.0
            } else {
                2.0_f64.sqrt()
            };
            let deviation = gain / (layer.inputs as f64).sqrt();
            for _ in 0..layer.parameters() {
                let radius = (-2.0 * (1.0 - f64::from(random.unit())).ln()).sqrt();
                let angle = std::f64::consts::TAU * f64::from(random.unit());
                parameters.push((deviation * radius * angle.cos()) as f32);
            }
            parameters.extend(std::iter::repeat_n(0.0, layer.outputs));
        }
        parameters
    }

    /// Preserve the Kaiming matrices and vary only the bias initialization.
    /// Domain mode changes the first layer; all later biases match uniform mode.
    pub fn initial_parameters_with_biases(
        self,
        seed: u64,
        mode: RegressionBiasInitialization,
        scale: f32,
        bounds: [[f32; 2]; 2],
    ) -> Result<Vec<f32>> {
        self.validate()?;
        if !scale.is_finite()
            || scale < 0.0
            || bounds
                .iter()
                .any(|range| range.iter().any(|value| !value.is_finite()) || range[0] > range[1])
        {
            return Err("bias initialization needs a finite nonnegative scale and ordered finite operand bounds".into());
        }
        let mut parameters = self.initial_parameters(seed);
        if mode == RegressionBiasInitialization::Zero {
            return Ok(parameters);
        }
        let mut offset = 0;
        for (index, layer) in self.layers().iter().enumerate() {
            // Layer-specific streams keep later biases paired across modes.
            let mut random = Random::new(
                seed ^ 0x62696173696e6974 ^ (index as u64).wrapping_mul(0x9e3779b97f4a7c15),
            );
            for channel in 0..layer.outputs {
                let bias = if index == 0 && mode == RegressionBiasInitialization::Domain {
                    let a =
                        bounds[0][0] + (0.05 + 0.9 * random.unit()) * (bounds[0][1] - bounds[0][0]);
                    let b =
                        bounds[1][0] + (0.05 + 0.9 * random.unit()) * (bounds[1][1] - bounds[1][0]);
                    -scale
                        * (parameters[offset + channel] * a
                            + parameters[offset + layer.outputs + channel] * b)
                } else {
                    scale * (2.0 * random.unit() - 1.0) / (layer.inputs as f32).sqrt()
                };
                parameters[offset + layer.parameters() + channel] = bias;
            }
            offset += layer.parameters() + layer.outputs;
        }
        validate_finite(&parameters, "initialized regression parameters")?;
        Ok(parameters)
    }
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod scalar_kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, width >= 1, linear.len() == out.len(), skip.len() == out.len(),
        out.len() == rows * width, bias.len() == width))]
    pub fn affine_activation<const S: usize>(
        rows: usize,
        width: usize,
        relu: bool,
        residual: bool,
        linear: &[f32],
        bias: &[[f32; 3]],
        skip: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = (S, rows);
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let value = linear[item] + bias[item % width][0];
            let activated = if relu && value < 0.0 { 0.0 } else { value };
            *output = if residual {
                skip[item] + activated
            } else {
                activated
            };
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, width >= 1, linear.len() == out.len(), gradient.len() == out.len(),
        out.len() == rows * width, bias.len() == width))]
    pub fn relu_gradient<const S: usize>(
        rows: usize,
        width: usize,
        linear: &[f32],
        bias: &[[f32; 3]],
        gradient: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = (S, rows);
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            *output = if linear[item] + bias[item % width][0] > 0.0 {
                gradient[item]
            } else {
                0.0
            };
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, width >= 1, gradient.len() == rows * width, out.len() == width))]
    pub fn bias_gradient<const S: usize>(
        rows: usize,
        width: usize,
        gradient: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let channel = index.get();
        if let Some(output) = out.get_mut(index) {
            let mut total = 0.0;
            for row in 0..rows {
                total += gradient[row * width + channel];
            }
            *output = total;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, offset + rows <= indices.len(), predictions.len() == rows, out.len() == rows))]
    pub fn mse_gradient<const S: usize>(
        rows: usize,
        offset: usize,
        indices: &[u32],
        targets: &[f32],
        predictions: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            *output =
                2.0 * (predictions[row] - targets[indices[offset + row] as usize]) / rows as f32;
        }
    }
}

pub struct RegressionTable {
    features: DeviceBuffer<f32>,
    targets: DeviceBuffer<f32>,
    identity: DeviceBuffer<u32>,
    rows: usize,
}

impl RegressionTable {
    pub fn rows(&self) -> usize {
        self.rows
    }
}

struct Workspace {
    rows: usize,
    inputs: DeviceBuffer<f32>,
    linear: Vec<DeviceBuffer<f32>>,
    hidden: Vec<DeviceBuffer<f32>>,
    normalized: Vec<DeviceBuffer<f32>>,
    gradients: Vec<DeviceBuffer<f32>>,
    output_linear: DeviceBuffer<f32>,
    predictions: DeviceBuffer<f32>,
    output_gradient: DeviceBuffer<f32>,
    statistics: DeviceBuffer<[f32; 2]>,
}

impl Workspace {
    fn new(model: &GpuRegressionMlp, rows: usize) -> Result<Self> {
        let (stream, config) = (&model.kernels.stream, model.config);
        let buffers = |count| {
            (0..count)
                .map(|_| DeviceBuffer::zeroed(stream, rows * config.width))
                .collect::<std::result::Result<Vec<_>, _>>()
        };
        Ok(Self {
            rows,
            inputs: DeviceBuffer::zeroed(stream, rows * 2)?,
            linear: buffers(config.depth)?,
            hidden: buffers(config.depth)?,
            normalized: buffers(if config.pre_rmsnorm {
                config.depth - 1
            } else {
                0
            })?,
            gradients: buffers(5)?,
            output_linear: DeviceBuffer::zeroed(stream, rows)?,
            predictions: DeviceBuffer::zeroed(stream, rows)?,
            output_gradient: DeviceBuffer::zeroed(stream, rows)?,
            statistics: DeviceBuffer::zeroed(stream, rows)?,
        })
    }
}

pub struct GpuRegressionMlp {
    config: RegressionConfig,
    kernels: Kernels,
    scalar: scalar_kernels::LoadedModule,
    parameters: Vec<Parameters>,
    workspaces: Vec<Workspace>,
    beta1_power: f32,
    beta2_power: f32,
}

impl GpuRegressionMlp {
    pub fn new(device: usize, config: RegressionConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        Self::from_parameters(device, config, &config.initial_parameters(seed))
    }

    /// Upload finite weights in the same layer/matrix/bias order as `parameters`.
    /// AdamW moments and its update counter start at zero.
    pub fn from_parameters(
        device: usize,
        config: RegressionConfig,
        initial: &[f32],
    ) -> Result<Self> {
        config.validate_parameters(initial)?;
        let context = CudaContext::new(device)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let tiled = dense_tiled::Kernels::new(&context, &stream)?;
        let scalar = scalar_kernels::load(&context)?;
        let mut offset = 0;
        let mut parameters = Vec::new();
        for layer in config.layers() {
            for size in [layer.parameters(), layer.outputs] {
                let state: Vec<_> = initial[offset..offset + size]
                    .iter()
                    .map(|&value| [value, 0.0, 0.0])
                    .collect();
                parameters.push(Parameters {
                    state: DeviceBuffer::from_host(&stream, &state)?,
                    gradient: DeviceBuffer::zeroed(&stream, size)?,
                });
                offset += size;
            }
        }
        Ok(Self {
            config,
            kernels: Kernels {
                stream,
                module,
                tiled,
                launches: Launches::default(),
            },
            scalar,
            parameters,
            workspaces: Vec::new(),
            beta1_power: 1.0,
            beta2_power: 1.0,
        })
    }

    pub fn upload_table(&self, features: &[f32], targets: &[f32]) -> Result<RegressionTable> {
        let rows = targets.len();
        if !(1..=MAX_SIMPLE_ROWS).contains(&rows) || features.len() != 2 * rows {
            return Err("regression tables need two features and one target per row".into());
        }
        validate_finite(features, "regression features")?;
        validate_finite(targets, "regression targets")?;
        let stream = &self.kernels.stream;
        Ok(RegressionTable {
            features: DeviceBuffer::from_host(stream, features)?,
            targets: DeviceBuffer::from_host(stream, targets)?,
            identity: DeviceBuffer::from_host(stream, &(0..rows as u32).collect::<Vec<_>>())?,
            rows,
        })
    }

    pub fn upload_batches(
        &self,
        table: &RegressionTable,
        indices: &[u32],
    ) -> Result<DeviceBatches> {
        if indices.is_empty()
            || indices.len() > MAX_SIMPLE_ROWS
            || indices.iter().any(|&index| index as usize >= table.rows)
        {
            return Err("invalid regression batch indices".into());
        }
        Ok(DeviceBatches {
            indices: DeviceBuffer::from_host(&self.kernels.stream, indices)?,
            table_rows: table.rows,
        })
    }

    pub fn update_batches(
        &self,
        table: &RegressionTable,
        batches: &mut DeviceBatches,
        indices: &[u32],
    ) -> Result<()> {
        if indices.is_empty()
            || indices.len() > MAX_SIMPLE_ROWS
            || indices.iter().any(|&index| index as usize >= table.rows)
        {
            return Err("invalid regression batch indices".into());
        }
        if batches.len() != indices.len() || batches.table_rows != table.rows {
            *batches = self.upload_batches(table, indices)?;
        } else {
            batches
                .indices
                .copy_from_host(&self.kernels.stream, indices)?;
        }
        Ok(())
    }

    fn take_workspace(&mut self, rows: usize) -> Result<Workspace> {
        if let Some(index) = self
            .workspaces
            .iter()
            .position(|workspace| workspace.rows == rows)
        {
            return Ok(self.workspaces.remove(index));
        }
        if self.workspaces.len() >= 4 {
            self.workspaces.remove(0);
        }
        Workspace::new(self, rows)
    }

    fn gather(
        &self,
        table: &RegressionTable,
        indices: &DeviceBuffer<u32>,
        offset: usize,
        workspace: &mut Workspace,
    ) -> Result<()> {
        let k = &self.kernels;
        let launch = k.launches.get(config::<1>(workspace.inputs.len())?, |c| {
            k.module.prepare_simple_gather::<0>(c)
        })?;
        k.module.simple_gather(
            &k.stream,
            &launch,
            2,
            workspace.rows,
            offset,
            indices,
            &table.features,
            &mut workspace.inputs,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn activate(
        &self,
        width: usize,
        relu: bool,
        residual: bool,
        linear: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<[f32; 3]>,
        skip: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let k = &self.kernels;
        let launch = k.launches.get(config::<1>(out.len())?, |c| {
            self.scalar.prepare_affine_activation::<0>(c)
        })?;
        self.scalar.affine_activation(
            &k.stream,
            &launch,
            linear.len() / width,
            width,
            relu,
            residual,
            linear,
            bias,
            skip,
            out,
        )?;
        Ok(())
    }

    fn forward(&self, workspace: &mut Workspace) -> Result<()> {
        let (k, c, rows) = (&self.kernels, self.config, workspace.rows);
        let layers = c.layers();
        for (index, layer) in layers[..c.depth].iter().enumerate() {
            let input = if index == 0 {
                &workspace.inputs
            } else if c.pre_rmsnorm {
                let hidden = &workspace.hidden[index - 1];
                k.rmsnorm::<0>(
                    rows,
                    c.width,
                    [hidden, hidden, hidden],
                    &mut workspace.statistics,
                    &mut workspace.normalized[index - 1],
                )?;
                &workspace.normalized[index - 1]
            } else {
                &workspace.hidden[index - 1]
            };
            k.dense::<IDENTITY>(
                *layer,
                rows,
                input,
                [layer.inputs, 0],
                &self.parameters[2 * index].state,
                &mut workspace.linear[index],
            )?;
            // Hidden output and previous skip must be disjoint borrows.
            if index == 0 {
                self.activate(
                    c.width,
                    true,
                    false,
                    &workspace.linear[index],
                    &self.parameters[2 * index + 1].state,
                    &workspace.linear[index],
                    &mut workspace.hidden[index],
                )?;
            } else {
                let [previous, output] = workspace.hidden.get_disjoint_mut([index - 1, index])?;
                self.activate(
                    c.width,
                    true,
                    c.residual,
                    &workspace.linear[index],
                    &self.parameters[2 * index + 1].state,
                    previous,
                    output,
                )?;
            }
        }
        let index = c.depth;
        k.dense::<IDENTITY>(
            layers[index],
            rows,
            &workspace.hidden[index - 1],
            [c.width, 0],
            &self.parameters[2 * index].state,
            &mut workspace.output_linear,
        )?;
        self.activate(
            1,
            false,
            false,
            &workspace.output_linear,
            &self.parameters[2 * index + 1].state,
            &workspace.output_linear,
            &mut workspace.predictions,
        )
    }

    fn bias_gradient(
        &mut self,
        index: usize,
        rows: usize,
        width: usize,
        gradient: &DeviceBuffer<f32>,
    ) -> Result<()> {
        let k = &self.kernels;
        let launch = k.launches.get(config::<1>(width)?, |c| {
            self.scalar.prepare_bias_gradient::<0>(c)
        })?;
        self.scalar.bias_gradient(
            &k.stream,
            &launch,
            rows,
            width,
            gradient,
            &mut self.parameters[2 * index + 1].gradient,
        )?;
        Ok(())
    }

    fn backward(&mut self, workspace: &mut Workspace) -> Result<()> {
        let (c, rows) = (self.config, workspace.rows);
        let layers = c.layers();
        let last = c.depth;
        self.kernels.weight_gradient::<IDENTITY>(
            layers[last],
            rows,
            &workspace.hidden[last - 1],
            [c.width, 0],
            &workspace.output_gradient,
            &mut self.parameters[2 * last].gradient,
        )?;
        self.bias_gradient(last, rows, 1, &workspace.output_gradient)?;
        self.kernels.input_gradient::<IDENTITY>(
            layers[last],
            rows,
            &workspace.output_gradient,
            &self.parameters[2 * last].state,
            &workspace.hidden[last - 1],
            &mut workspace.gradients[0],
        )?;
        for index in (0..c.depth).rev() {
            let k = &self.kernels;
            let [upstream, branch] = workspace.gradients.get_disjoint_mut([0, 1])?;
            let launch = k.launches.get(config::<1>(branch.len())?, |c| {
                self.scalar.prepare_relu_gradient::<0>(c)
            })?;
            self.scalar.relu_gradient(
                &k.stream,
                &launch,
                rows,
                c.width,
                &workspace.linear[index],
                &self.parameters[2 * index + 1].state,
                upstream,
                branch,
            )?;
            let input = if index == 0 {
                &workspace.inputs
            } else if c.pre_rmsnorm {
                &workspace.normalized[index - 1]
            } else {
                &workspace.hidden[index - 1]
            };
            k.weight_gradient::<IDENTITY>(
                layers[index],
                rows,
                input,
                [layers[index].inputs, 0],
                &workspace.gradients[1],
                &mut self.parameters[2 * index].gradient,
            )?;
            self.bias_gradient(index, rows, c.width, &workspace.gradients[1])?;
            if index == 0 {
                break;
            }
            let k = &self.kernels;
            let [branch, normalized] = workspace.gradients.get_disjoint_mut([1, 2])?;
            k.input_gradient::<IDENTITY>(
                layers[index],
                rows,
                branch,
                &self.parameters[2 * index].state,
                input,
                normalized,
            )?;
            let source = if c.pre_rmsnorm {
                let [normalized, out] = workspace.gradients.get_disjoint_mut([2, 3])?;
                k.rmsnorm::<1>(
                    rows,
                    c.width,
                    [
                        &workspace.hidden[index - 1],
                        normalized,
                        &workspace.normalized[index - 1],
                    ],
                    &mut workspace.statistics,
                    out,
                )?;
                3
            } else {
                2
            };
            if c.residual {
                let [skip, branch, out] = workspace.gradients.get_disjoint_mut([0, source, 4])?;
                k.combine::<ADD>(skip, branch, out)?;
                workspace.gradients.swap(0, 4);
            } else {
                workspace.gradients.swap(0, source);
            }
        }
        Ok(())
    }

    fn gradients(
        &mut self,
        table: &RegressionTable,
        batches: &DeviceBatches,
        offset: usize,
        rows: usize,
        read_predictions: bool,
    ) -> Result<Vec<f32>> {
        if rows == 0
            || rows > MAX_SIMPLE_ROWS
            || batches.table_rows != table.rows
            || offset
                .checked_add(rows)
                .is_none_or(|end| end > batches.len())
        {
            return Err("invalid regression minibatch range or table".into());
        }
        let mut workspace = self.take_workspace(rows)?;
        let result = (|| {
            self.gather(table, &batches.indices, offset, &mut workspace)?;
            self.forward(&mut workspace)?;
            let k = &self.kernels;
            let launch = k.launches.get(config::<1>(rows)?, |c| {
                self.scalar.prepare_mse_gradient::<0>(c)
            })?;
            self.scalar.mse_gradient(
                &k.stream,
                &launch,
                rows,
                offset,
                &batches.indices,
                &table.targets,
                &workspace.predictions,
                &mut workspace.output_gradient,
            )?;
            self.backward(&mut workspace)?;
            if read_predictions {
                Ok(workspace.predictions.to_host_vec(&self.kernels.stream)?)
            } else {
                Ok(Vec::new())
            }
        })();
        self.workspaces.push(workspace);
        result
    }

    pub fn train_batch(
        &mut self,
        table: &RegressionTable,
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

    pub fn predictions(&mut self, table: &RegressionTable) -> Result<Vec<f32>> {
        let mut workspace = self.take_workspace(table.rows)?;
        let result = (|| {
            self.gather(table, &table.identity, 0, &mut workspace)?;
            self.forward(&mut workspace)?;
            let predictions = workspace.predictions.to_host_vec(&self.kernels.stream)?;
            validate_finite(&predictions, "regression predictions")?;
            Ok(predictions)
        })();
        self.workspaces.push(workspace);
        result
    }

    pub fn parameters(&self) -> Result<Vec<f32>> {
        let mut parameters = Vec::with_capacity(self.config.parameter_count());
        for group in &self.parameters {
            parameters.extend(
                group
                    .state
                    .to_host_vec(&self.kernels.stream)?
                    .iter()
                    .map(|state| state[0]),
            );
        }
        validate_finite(&parameters, "regression parameters")?;
        Ok(parameters)
    }

    pub fn loss_and_gradient(
        &mut self,
        table: &RegressionTable,
        indices: &[u32],
    ) -> Result<(f64, Vec<f64>)> {
        let batches = self.upload_batches(table, indices)?;
        let predictions = self.gradients(table, &batches, 0, indices.len(), true)?;
        let targets = table.targets.to_host_vec(&self.kernels.stream)?;
        let loss = predictions
            .iter()
            .zip(indices)
            .map(|(&prediction, &index)| {
                (f64::from(prediction) - f64::from(targets[index as usize])).powi(2)
            })
            .sum::<f64>()
            / indices.len() as f64;
        let mut gradients = Vec::with_capacity(self.config.parameter_count());
        for group in &self.parameters {
            gradients.extend(
                group
                    .gradient
                    .to_host_vec(&self.kernels.stream)?
                    .into_iter()
                    .map(f64::from),
            );
        }
        if !loss.is_finite() || gradients.iter().any(|value| !value.is_finite()) {
            return Err("nonfinite regression loss or gradient".into());
        }
        Ok((loss, gradients))
    }

    /// Outputs, every parameter derivative, tail/offset batches and AdamW.
    pub fn verify(device: usize) -> Result<()> {
        for width in [3, 16] {
            for depth in [1, 3] {
                for pre_rmsnorm in [false, true] {
                    for residual in [false, true] {
                        if depth == 1 && (pre_rmsnorm || residual) {
                            continue;
                        }
                        for input_scale in [1.0, 96.0] {
                            let config = RegressionConfig {
                                width,
                                depth,
                                pre_rmsnorm,
                                residual,
                            };
                            let mut model = Self::new(device, config, 11)?;
                            let parameters = model.parameters()?;
                            let features: Vec<_> = (0..37)
                                .flat_map(|row| {
                                    [
                                        if row % 7 == 0 {
                                            0.0
                                        } else {
                                            input_scale * (0.1 + row as f32 * 0.021)
                                        },
                                        if row % 5 == 0 {
                                            0.0
                                        } else {
                                            input_scale * (0.9 - row as f32 * 0.013)
                                        },
                                    ]
                                })
                                .collect();
                            let targets: Vec<_> = (0..37)
                                .map(|row| input_scale * (row * 7 % 19) as f32 / 18.0)
                                .collect();
                            let table = model.upload_table(&features, &targets)?;
                            if model.upload_table(&[0.0], &[0.0]).is_ok()
                                || model.upload_batches(&table, &[]).is_ok()
                                || model.upload_batches(&table, &[37]).is_ok()
                            {
                                return Err("invalid regression data was accepted".into());
                            }
                            let actual = model.predictions(&table)?;
                            let expected = reference(config, &parameters, &features, &targets)?.0;
                            compare(
                                "regression predictions",
                                &expected,
                                &actual.iter().copied().map(f64::from).collect::<Vec<_>>(),
                            )?;
                            for indices in [vec![2], vec![12, 3, 12, 8, 0], (0..37).rev().collect()]
                            {
                                let selected_features: Vec<_> = indices
                                    .iter()
                                    .flat_map(|&index| {
                                        features[2 * index as usize..2 * index as usize + 2]
                                            .iter()
                                            .copied()
                                    })
                                    .collect();
                                let selected_targets: Vec<_> = indices
                                    .iter()
                                    .map(|&index| targets[index as usize])
                                    .collect();
                                let (_, loss, expected) = reference(
                                    config,
                                    &parameters,
                                    &selected_features,
                                    &selected_targets,
                                )?;
                                let (actual_loss, actual) =
                                    model.loss_and_gradient(&table, &indices)?;
                                compare("regression MSE", &[loss], &[actual_loss])?;
                                compare("regression gradients", &expected, &actual)?;
                            }
                            let indices = [5, 2, 17, 9, 3];
                            let batches = model.upload_batches(&table, &indices)?;
                            let (_, gradients) = model.loss_and_gradient(&table, &indices[1..4])?;
                            let optimizer = AdamWOptions {
                                learning_rate: 0.003,
                                weight_decay: 0.1,
                                beta2: 0.98,
                                ..AdamWOptions::default()
                            };
                            model.train_batch(&table, &batches, 1, 3, optimizer)?;
                            let actual = model.parameters()?;
                            let expected: Vec<_> = parameters
                                .iter()
                                .zip(gradients)
                                .map(|(&parameter, gradient)| {
                                    let gradient = gradient as f32;
                                    f64::from(
                                        parameter
                                            - optimizer.learning_rate
                                                * (gradient / (gradient.abs() + optimizer.epsilon)
                                                    + optimizer.weight_decay * parameter),
                                    )
                                })
                                .collect();
                            compare(
                                "regression AdamW",
                                &expected,
                                &actual.iter().copied().map(f64::from).collect::<Vec<_>>(),
                            )?;
                            println!(
                                "PASS scalar MLP width={width} depth={depth} pre_rmsnorm={pre_rmsnorm} residual={residual} input_scale={input_scale}: f64 outputs, MSE, gradients, gathered rows, offset AdamW and input validation"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn compare(name: &str, expected: &[f64], actual: &[f64]) -> Result<()> {
    let scale = expected
        .iter()
        .fold(1.0_f64, |scale, value| scale.max(value.abs()));
    if expected.len() != actual.len()
        || expected.iter().zip(actual).any(|(expected, actual)| {
            !actual.is_finite() || (actual - expected).abs() > scale * 3e-5
        })
    {
        return Err(format!("{name} differs from independent f64 reference").into());
    }
    Ok(())
}

fn reference(
    config: RegressionConfig,
    parameters: &[f32],
    features: &[f32],
    targets: &[f32],
) -> Result<(Vec<f64>, f64, Vec<f64>)> {
    let mut tape = Tape::new(parameters.iter().copied().map(f64::from));
    let layers = config.layers();
    let mut predictions = Vec::new();
    let mut losses = Vec::new();
    for (row, &target) in features.as_chunks::<2>().0.iter().zip(targets) {
        let mut hidden: Vec<_> = row
            .iter()
            .map(|&value| tape.constant(f64::from(value)))
            .collect();
        let mut offset = 0;
        for (index, layer) in layers.iter().enumerate() {
            let input = if index > 0 && index < config.depth && config.pre_rmsnorm {
                rmsnorm(&mut tape, std::slice::from_ref(&hidden), 1).remove(0)
            } else {
                hidden.clone()
            };
            let next = (0..layer.outputs)
                .map(|channel| {
                    let weights: Vec<_> = (0..layer.inputs)
                        .map(|input| offset + input * layer.outputs + channel)
                        .collect();
                    let affine = tape.dot(&input, &weights);
                    let biased = tape.add(affine, offset + layer.parameters() + channel);
                    let value = if index < config.depth {
                        tape.relu(biased)
                    } else {
                        biased
                    };
                    if index > 0 && index < config.depth && config.residual {
                        tape.add(hidden[channel], value)
                    } else {
                        value
                    }
                })
                .collect();
            hidden = next;
            offset += layer.parameters() + layer.outputs;
        }
        let prediction = hidden[0];
        let target = tape.constant(f64::from(target));
        let difference = tape.subtract(prediction, target);
        losses.push(tape.multiply(difference, difference));
        predictions.push(tape.values[prediction]);
    }
    let sum = tape.sum(&losses);
    let loss = tape.scale(sum, 1.0 / targets.len() as f64);
    Ok((
        predictions,
        tape.values[loss],
        tape.backward(loss, parameters.len()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_parameter_validation_rejects_invalid_initializers() {
        let config = RegressionConfig {
            width: 3,
            depth: 1,
            pre_rmsnorm: false,
            residual: false,
        };
        let error = config.validate_parameters(&[]).unwrap_err();
        assert!(error.to_string().contains("parameter count"));
        let mut parameters = config.initial_parameters(11);
        parameters[0] = f32::NAN;
        let error = config.validate_parameters(&parameters).unwrap_err();
        assert!(error.to_string().contains("finite"));
        for scale in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(
                config
                    .initial_parameters_with_biases(
                        11,
                        RegressionBiasInitialization::Uniform,
                        scale,
                        [[0.0, 1.0]; 2]
                    )
                    .is_err()
            );
        }
        for bounds in [[[1.0, 0.0]; 2], [[0.0, f32::NAN]; 2]] {
            assert!(
                config
                    .initial_parameters_with_biases(
                        11,
                        RegressionBiasInitialization::Domain,
                        1.0,
                        bounds
                    )
                    .is_err()
            );
        }
    }
    #[test]
    fn regression_initialization_preserves_paired_controls_and_zero_biases() {
        let base = RegressionConfig {
            width: 8,
            depth: 3,
            pre_rmsnorm: false,
            residual: false,
        };
        let parameters = base.initial_parameters(11);
        assert_eq!(parameters.len(), base.parameter_count());
        for pre_rmsnorm in [false, true] {
            for residual in [false, true] {
                assert_eq!(
                    parameters,
                    RegressionConfig {
                        pre_rmsnorm,
                        residual,
                        ..base
                    }
                    .initial_parameters(11)
                );
            }
        }
        let mut offset = 0;
        for layer in base.layers() {
            offset += layer.parameters();
            assert!(
                parameters[offset..offset + layer.outputs]
                    .iter()
                    .all(|&value| value == 0.0)
            );
            offset += layer.outputs;
        }
    }
    #[test]
    fn regression_reference_mse_matches_finite_differences() {
        for pre_rmsnorm in [false, true] {
            for residual in [false, true] {
                let config = RegressionConfig {
                    width: 3,
                    depth: 2,
                    pre_rmsnorm,
                    residual,
                };
                let parameters = config.initial_parameters(13);
                let features = [0.31, 0.82, 0.72, 0.19];
                let targets = [0.2, 0.7];
                let (_, _, gradient) = reference(config, &parameters, &features, &targets).unwrap();
                for (index, &expected) in gradient.iter().enumerate() {
                    let mut plus = parameters.clone();
                    let mut minus = parameters.clone();
                    plus[index] += 1e-3;
                    minus[index] -= 1e-3;
                    let delta = f64::from(plus[index]) - f64::from(minus[index]);
                    let approximate = (reference(config, &plus, &features, &targets).unwrap().1
                        - reference(config, &minus, &features, &targets).unwrap().1)
                        / delta;
                    assert!(
                        (expected - approximate).abs() < 3e-3 * expected.abs().max(1.0),
                        "parameter {index}: {expected} vs {approximate}"
                    );
                }
            }
        }
    }
}
