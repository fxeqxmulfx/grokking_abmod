//! A smaller trainable numerical vocabulary for the existing tied Dense model.
//! Both operand lookup and the 97-class readout use the same generated table.
use super::{DeviceBatches, DeviceTable, Evaluation, GpuSimpleModel, Parameters, config};
use crate::{
    Result,
    gpt::AdamWOptions,
    mesh::validate_finite,
    simple::{Architecture, MAX_SIMPLE_CLASSES, MAX_SIMPLE_WIDTH, SimpleConfig},
};
use cuda_core::DeviceBuffer;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

/// Exact integer routing and fixed RoPE phases, shared by input and output.
#[derive(Clone, Debug)]
pub struct AnchorEncoding {
    anchors: Vec<usize>,
    assignments: Vec<u32>,
    phases: Vec<f32>,
    width: usize,
}

impl AnchorEncoding {
    pub fn new(modulus: usize, width: usize, anchors: Vec<usize>, rotate: bool) -> Result<Self> {
        if !(2..=MAX_SIMPLE_CLASSES).contains(&modulus)
            || !(2..=MAX_SIMPLE_WIDTH).contains(&width)
            || !width.is_multiple_of(2)
            || anchors.is_empty()
            || anchors.iter().any(|&q| q >= modulus)
            || anchors.windows(2).any(|q| q[0] >= q[1])
        {
            return Err(
                "anchors must be distinct ordered residues with an even embedding width".into(),
            );
        }
        let mut assignments = Vec::with_capacity(modulus);
        let mut phases = Vec::with_capacity(modulus * width);
        for value in 0..modulus {
            // Fixed ordinary distance between canonical residues; ties choose the lower anchor.
            let i = anchors
                .iter()
                .enumerate()
                .min_by_key(|&(_, &q)| value.abs_diff(q))
                .ok_or("empty anchor dictionary")?
                .0;
            assignments.push(i as u32);
            let displacement = value as f64 - anchors[i] as f64;
            for pair in 0..width / 2 {
                let frequency = 10000.0_f64.powf(-((2 * pair) as f64) / width as f64);
                let angle = if rotate {
                    frequency * displacement
                } else {
                    0.0
                };
                phases.extend([angle.cos() as f32, angle.sin() as f32]);
            }
        }
        Ok(Self {
            anchors,
            assignments,
            phases,
            width,
        })
    }

    pub fn anchors(&self) -> &[usize] {
        &self.anchors
    }
    pub fn anchor_for(&self, value: usize) -> Option<usize> {
        self.assignments
            .get(value)
            .map(|&i| self.anchors[i as usize])
    }

    fn expand(&self, reduced: &[f32]) -> Vec<f32> {
        let mut result =
            Vec::with_capacity(self.phases.len() + reduced.len() - self.anchors.len() * self.width);
        for (value, &anchor) in self.assignments.iter().enumerate() {
            for pair in 0..self.width / 2 {
                let base = anchor as usize * self.width + 2 * pair;
                let [cosine, sine] = [
                    self.phases[value * self.width + 2 * pair],
                    self.phases[value * self.width + 2 * pair + 1],
                ];
                result.extend([
                    cosine * reduced[base] - sine * reduced[base + 1],
                    sine * reduced[base] + cosine * reduced[base + 1],
                ]);
            }
        }
        result.extend_from_slice(&reduced[self.anchors.len() * self.width..]);
        result
    }
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod anchor_kernels {
    use super::*;

    /// Materialized rows are derived values, with no independent optimizer state.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        classes >= 2, anchors >= 1, pairs >= 1, width == 2 * pairs,
        assignments.len() == classes, phases.len() == classes * width,
        state.len() == anchors * width, out.len() == classes * width))]
    pub fn generate<const S: usize>(
        classes: usize,
        anchors: usize,
        pairs: usize,
        width: usize,
        assignments: &[u32],
        phases: &[f32],
        state: &[[f32; 3]],
        mut out: DisjointSlice<[f32; 3]>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let value = item / width;
            let channel = item % width;
            let pair = channel / 2 * 2;
            let base = assignments[value] as usize * width + pair;
            let cosine = phases[value * width + pair];
            let sine = phases[value * width + pair + 1];
            let weight = if channel.is_multiple_of(2) {
                cosine * state[base][0] - sine * state[base + 1][0]
            } else {
                sine * state[base][0] + cosine * state[base + 1][0]
            };
            *output = [weight, 0.0, 0.0];
        }
        let _ = (classes, anchors, pairs);
    }

    /// The existing model accumulates input and tied-output gradients into dE.
    /// Each anchor parameter then owns the complete sum of R^T dE over its region.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        classes >= 2, anchors >= 1, pairs >= 1, width == 2 * pairs,
        assignments.len() == classes, phases.len() == classes * width,
        gradient.len() == classes * width, out.len() == anchors * width))]
    pub fn reduce_gradient<const S: usize>(
        classes: usize,
        anchors: usize,
        pairs: usize,
        width: usize,
        assignments: &[u32],
        phases: &[f32],
        gradient: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let anchor = item / width;
            let channel = item % width;
            let pair = channel / 2 * 2;
            let mut sum = 0.0;
            let mut value = 0;
            while value < classes {
                if assignments[value] as usize == anchor {
                    let base = value * width + pair;
                    let cosine = phases[base];
                    let sine = phases[base + 1];
                    sum += if channel.is_multiple_of(2) {
                        cosine * gradient[base] + sine * gradient[base + 1]
                    } else {
                        -sine * gradient[base] + cosine * gradient[base + 1]
                    };
                }
                value += 1;
            }
            *output = sum;
        }
        let _ = (anchors, pairs);
    }
}

/// Wrapper that trains only K anchor rows and the unchanged Dense core on CUDA.
pub struct GpuAnchoredModel {
    model: GpuSimpleModel,
    encoding: AnchorEncoding,
    anchors: Parameters,
    assignments: DeviceBuffer<u32>,
    phases: DeviceBuffer<f32>,
    module: anchor_kernels::LoadedModule,
}

impl GpuAnchoredModel {
    pub fn new(
        device: usize,
        config: SimpleConfig,
        encoding: AnchorEncoding,
        seed: u64,
    ) -> Result<Self> {
        if config.architecture != Architecture::TiedRotaryResidualDenseMlp
            || config.modulus != encoding.assignments.len()
            || config.width != encoding.width
        {
            return Err("anchor encoding needs a matching tied rotary Dense configuration".into());
        }
        let model = GpuSimpleModel::new(device, config, seed)?;
        let initial = config.initial_parameters(seed);
        let weights: Vec<_> = encoding
            .anchors
            .iter()
            .flat_map(|&q| {
                initial[q * config.width..(q + 1) * config.width]
                    .iter()
                    .map(|&p| [p, 0.0, 0.0])
            })
            .collect();
        let stream = &model.kernels.stream;
        let result = Self {
            anchors: Parameters {
                state: DeviceBuffer::from_host(stream, &weights)?,
                gradient: DeviceBuffer::zeroed(stream, weights.len())?,
            },
            assignments: DeviceBuffer::from_host(stream, &encoding.assignments)?,
            phases: DeviceBuffer::from_host(stream, &encoding.phases)?,
            module: anchor_kernels::load(stream.context())?,
            model,
            encoding,
        };
        Ok(result)
    }

    pub fn parameter_count(&self) -> usize {
        let c = self.model.config;
        c.parameter_count() - c.modulus * c.width + self.anchors.state.len()
    }

    fn generate(&mut self) -> Result<()> {
        let c = self.model.config;
        let launch = self
            .model
            .kernels
            .launches
            .get(config::<1>(c.modulus * c.width)?, |l| {
                self.module.prepare_generate::<0>(l)
            })?;
        self.module.generate(
            &self.model.kernels.stream,
            &launch,
            c.modulus,
            self.encoding.anchors.len(),
            c.width / 2,
            c.width,
            &self.assignments,
            &self.phases,
            &self.anchors.state,
            &mut self.model.parameters[0].state,
        )?;
        Ok(())
    }

    fn reduce_gradient(&mut self) -> Result<()> {
        let c = self.model.config;
        let launch = self
            .model
            .kernels
            .launches
            .get(config::<1>(self.anchors.gradient.len())?, |l| {
                self.module.prepare_reduce_gradient::<0>(l)
            })?;
        self.module.reduce_gradient(
            &self.model.kernels.stream,
            &launch,
            c.modulus,
            self.encoding.anchors.len(),
            c.width / 2,
            c.width,
            &self.assignments,
            &self.phases,
            &self.model.parameters[0].gradient,
            &mut self.anchors.gradient,
        )?;
        Ok(())
    }

    pub fn upload_table(&self, features: &[f32], targets: &[u32]) -> Result<DeviceTable> {
        self.model.upload_table(features, targets)
    }
    pub fn upload_batches(&self, table: &DeviceTable, indices: &[u32]) -> Result<DeviceBatches> {
        self.model.upload_batches(table, indices)
    }
    pub fn evaluate<const N: usize>(
        &mut self,
        tables: [&DeviceTable; N],
    ) -> Result<[Evaluation; N]> {
        self.generate()?;
        self.model.evaluate(tables)
    }
    pub fn reuse(&mut self, evaluation: Evaluation) {
        self.model.reuse(evaluation.logits);
    }

    pub fn train_batch(
        &mut self,
        table: &DeviceTable,
        batches: &DeviceBatches,
        offset: usize,
        rows: usize,
        optimizer: AdamWOptions,
    ) -> Result<()> {
        optimizer.validate()?;
        self.generate()?;
        self.model.gradients(table, batches, offset, rows, false)?;
        self.reduce_gradient()?;
        self.model.beta1_power *= optimizer.beta1;
        self.model.beta2_power *= optimizer.beta2;
        let corrections = [1.0 - self.model.beta1_power, 1.0 - self.model.beta2_power];
        self.model
            .kernels
            .adam(optimizer, corrections, &mut self.anchors)?;
        for group in self.model.parameters.iter_mut().skip(1) {
            self.model.kernels.adam(optimizer, corrections, group)?;
        }
        Ok(())
    }

    pub fn parameters(&mut self) -> Result<Vec<f32>> {
        let mut weights =
            DeviceBuffer::zeroed(&self.model.kernels.stream, self.anchors.state.len())?;
        self.model
            .kernels
            .weights(&self.anchors.state, &mut weights)?;
        let mut result = weights.to_host_vec(&self.model.kernels.stream)?;
        let full = self.model.parameters()?;
        result.extend_from_slice(&full[self.model.config.modulus * self.model.config.width..]);
        Ok(result)
    }

    fn loss_and_gradient(
        &mut self,
        table: &DeviceTable,
        indices: &[u32],
    ) -> Result<(f64, Vec<f64>)> {
        self.generate()?;
        let (loss, full) = self.model.loss_and_gradient(table, indices)?;
        self.reduce_gradient()?;
        let mut gradient: Vec<_> = self
            .anchors
            .gradient
            .to_host_vec(&self.model.kernels.stream)?
            .into_iter()
            .map(f64::from)
            .collect();
        gradient.extend_from_slice(&full[self.model.config.modulus * self.model.config.width..]);
        Ok((loss, gradient))
    }

    pub fn verify(device: usize) -> Result<()> {
        use crate::{
            gpt::{EquationDataset, ModularOperation},
            simple::{DenseEmbeddingInitialization, ReferenceSimpleModel},
        };
        let c = SimpleConfig {
            architecture: Architecture::TiedRotaryResidualDenseMlp,
            modulus: 7,
            width: 6,
            depth: 1,
            dense_embedding_initialization: DenseEmbeddingInitialization::NormalWidth,
            ..SimpleConfig::default()
        };
        let equations = EquationDataset::new(7, ModularOperation::Divide, 0.5, 11)?;
        let (features, targets) = c.encode(equations.split(false))?;
        for (anchors, rotate) in [
            (vec![0, 3, 6], true),
            (vec![0, 3, 6], false),
            ((0..7).collect(), true),
            (vec![0], true),
        ] {
            let encoding = AnchorEncoding::new(c.modulus, c.width, anchors, rotate)?;
            let mut model = Self::new(device, c, encoding.clone(), 11)?;
            let parameters = model.parameters()?;
            validate_finite(&parameters, "anchor verification parameters")?;
            let full = encoding.expand(&parameters);
            let reference = ReferenceSimpleModel::from_parameters(c, full)?;
            let table = model.upload_table(&features, &targets)?;
            let sampled = [2, 0, 3, 2, 1];
            let selected: Vec<_> = sampled[1..]
                .iter()
                .flat_map(|&r| {
                    features[r as usize * c.input_features()..(r as usize + 1) * c.input_features()]
                        .iter()
                        .copied()
                })
                .collect();
            let labels: Vec<_> = sampled[1..].iter().map(|&r| targets[r as usize]).collect();
            let (expected_loss, full_gradient) = reference.loss_and_gradient(&selected, &labels)?;
            let mut expected = vec![0.0; parameters.len()];
            for (value, &anchor) in encoding.assignments.iter().enumerate() {
                for pair in 0..c.width / 2 {
                    let base = value * c.width + 2 * pair;
                    let target = anchor as usize * c.width + 2 * pair;
                    let (cosine, sine) = (
                        f64::from(encoding.phases[base]),
                        f64::from(encoding.phases[base + 1]),
                    );
                    expected[target] +=
                        cosine * full_gradient[base] + sine * full_gradient[base + 1];
                    expected[target + 1] +=
                        -sine * full_gradient[base] + cosine * full_gradient[base + 1];
                }
            }
            let prefix = encoding.anchors.len() * c.width;
            expected[prefix..].copy_from_slice(&full_gradient[c.modulus * c.width..]);
            let (loss, gradient) = model.loss_and_gradient(&table, &sampled[1..])?;
            if (loss - expected_loss).abs() > 1e-5 {
                return Err("anchor reference loss mismatch".into());
            }
            for (i, (&actual, &wanted)) in gradient.iter().zip(&expected).enumerate() {
                if !actual.is_finite() || (actual - wanted).abs() > 1e-4 * (1.0 + wanted.abs()) {
                    return Err(format!("anchor gradient {i}: CUDA {actual}, f64 {wanted}").into());
                }
            }
            let optimizer = AdamWOptions {
                learning_rate: 0.001,
                weight_decay: 0.01,
                ..AdamWOptions::default()
            };
            let batches = model.upload_batches(&table, &sampled)?;
            model.train_batch(&table, &batches, 1, 4, optimizer)?;
            let updated = model.parameters()?;
            for (i, ((&p, &g), &actual)) in
                parameters.iter().zip(&gradient).zip(&updated).enumerate()
            {
                let g = g as f32;
                let m = (1.0 - optimizer.beta1) * g;
                let v = (1.0 - optimizer.beta2) * g * g;
                let wanted = p - optimizer.learning_rate
                    * (m / (1.0 - optimizer.beta1)
                        / ((v / (1.0 - optimizer.beta2)).sqrt() + optimizer.epsilon)
                        + optimizer.weight_decay * p);
                if (actual - wanted).abs() > 3e-6 {
                    return Err(format!("anchor AdamW mismatch at {i}").into());
                }
            }
            if encoding.anchors.len() == c.modulus {
                let mut plain = GpuSimpleModel::new(device, c, 11)?;
                let ordinary_table = plain.upload_table(&features, &targets)?;
                let ordinary_batches = plain.upload_batches(&ordinary_table, &sampled)?;
                plain.train_batch(&ordinary_table, &ordinary_batches, 1, 4, optimizer)?;
                let original = plain.parameters()?;
                for (&a, &b) in updated.iter().zip(&original) {
                    if a.to_bits() != b.to_bits() {
                        return Err("full anchor table differs from original CUDA model".into());
                    }
                }
                println!(
                    "PASS full anchor vocabulary: bitwise equivalence to the original CUDA model after AdamW"
                );
            }
            println!(
                "PASS anchored CUDA K={} rotate={rotate}: f64 loss/all gradients, tied input+output, offset/repeated batch, AdamW",
                encoding.anchors.len()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_routing_and_fixed_rotations_cover_every_residue() {
        let encoding = AnchorEncoding::new(97, 16, vec![0, 16, 32, 48, 64, 80, 96], true).unwrap();
        assert_eq!(encoding.anchor_for(8), Some(0));
        assert_eq!(encoding.anchor_for(9), Some(16));
        assert_eq!(encoding.anchor_for(96), Some(96));
        assert_eq!(encoding.anchor_for(97), None);
        for (value, &assignment) in encoding.assignments.iter().enumerate() {
            let anchor = encoding.anchors[assignment as usize];
            let distance = value.abs_diff(anchor);
            assert!(
                encoding
                    .anchors
                    .iter()
                    .all(|&q| distance <= value.abs_diff(q))
            );
            for pair in encoding.phases[value * 16..(value + 1) * 16]
                .as_chunks::<2>()
                .0
            {
                assert!((pair[0] * pair[0] + pair[1] * pair[1] - 1.0).abs() < 2e-7);
            }
        }
        assert!(AnchorEncoding::new(97, 16, vec![1, 1], true).is_err());
        assert!(AnchorEncoding::new(97, 15, vec![0], true).is_err());
        assert!(AnchorEncoding::new(97, 16, vec![97], true).is_err());
    }
}
