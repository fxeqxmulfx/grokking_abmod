//! Shared numerical encoders with an independent full-class readout. The dense
//! operand matrix is a derived cache, not a separately optimized parameter.
use super::{DeviceBatches, DeviceTable, Evaluation, GpuSimpleModel, Parameters, config};
use crate::{
    Result,
    gpt::AdamWOptions,
    mesh::{Random, validate_finite},
    simple::{Architecture, Encoding, SimpleConfig, discrete_logarithms},
};
use cuda_core::DeviceBuffer;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::collections::BTreeSet;

/// A fixed, exact-identity-preserving code for the finite modular domain.
#[derive(Clone, Debug)]
pub struct NumericalCode {
    name: String,
    classes: usize,
    features: usize,
    values: Vec<f32>,
}

impl NumericalCode {
    pub const NAMES: [&str; 10] = [
        "table",
        "bits",
        "digits",
        "fone",
        "binary-fourier",
        "fourier",
        "log-fourier",
        "random",
        "random-bits",
        "integer-bits",
    ];

    pub fn new(classes: usize, name: &str, harmonics: usize) -> Result<Self> {
        if !(2..=crate::simple::MAX_SIMPLE_CLASSES).contains(&classes)
            || !Self::NAMES.contains(&name)
            || !(1..=48).contains(&harmonics)
        {
            return Err("invalid numerical code settings".into());
        }
        if name == "random-bits" && (1_u128 << (2 * harmonics)) < classes as u128 {
            return Err(
                "random binary codes need enough distinct bit strings for every class".into(),
            );
        }
        let bits = Encoding::Binary.features(classes);
        let decimal_places = (classes - 1).ilog10() as usize + 1;
        let logs = if name == "log-fourier" {
            discrete_logarithms(classes)?
        } else {
            Vec::new()
        };
        let mut random = Random::new(0x6e756d636f646501);
        let mut binary_words = BTreeSet::new();
        let mut values = Vec::new();
        for x in 0..classes {
            let mut row = Vec::new();
            let mut circle = |position: f64, period: f64| {
                let angle = std::f64::consts::TAU * position / period;
                row.extend([angle.cos() as f32, angle.sin() as f32]);
            };
            match name {
                "table" => {
                    row.resize(classes, 0.0);
                    row[x] = 1.0;
                }
                "bits" => row.extend((0..bits).map(|b| if x >> b & 1 == 1 { 1.0 } else { -1.0 })),
                "integer-bits" => {
                    // Ordinary MSB-first positional digits, followed by a shared
                    // affine bias feature. The zero numeral remains representable.
                    row.extend((0..bits).rev().map(|b| ((x >> b) & 1) as f32));
                    row.push(1.0);
                }
                "digits" => {
                    row.resize(10 * decimal_places, 0.0);
                    for p in 0..decimal_places {
                        row[10 * p + x / 10_usize.pow(p as u32) % 10] = 1.0;
                    }
                }
                "fone" => {
                    for p in 1..=decimal_places {
                        circle(x as f64, 10.0_f64.powi(p as i32));
                    }
                }
                "binary-fourier" => {
                    for p in 1..=bits {
                        circle(x as f64, 2.0_f64.powi(p as i32));
                    }
                }
                "fourier" => {
                    for h in 1..=harmonics {
                        circle((h * x) as f64, classes as f64);
                    }
                }
                "log-fourier" => {
                    for h in 1..=harmonics {
                        circle((h * logs[x]) as f64, (classes - 1) as f64);
                    }
                    if x == 0 {
                        row.fill(0.0);
                    }
                    row.push(if x == 0 { 1.0 } else { 0.0 });
                }
                "random" => {
                    for _ in 0..2 * harmonics {
                        row.push(normal(&mut random));
                    }
                }
                "random-bits" => {
                    // Sample uniformly without replacement so identities never collide.
                    let word = loop {
                        let mut candidate = 0_u128;
                        for bit in 0..2 * harmonics {
                            candidate |= (random.index(2) as u128) << bit;
                        }
                        if binary_words.insert(candidate) {
                            break candidate;
                        }
                    };
                    row.extend(
                        (0..2 * harmonics).map(|bit| if word >> bit & 1 == 1 { 1.0 } else { -1.0 }),
                    );
                }
                _ => unreachable!(),
            }
            let norm = if name == "integer-bits" {
                // One global scale preserves the literal bit values and the
                // constant bias coefficient, unlike per-numeral normalization.
                (row.len() as f64).sqrt()
            } else {
                row.iter()
                    .map(|&v| f64::from(v).powi(2))
                    .sum::<f64>()
                    .sqrt()
            };
            values.extend(row.into_iter().map(|v| (f64::from(v) / norm) as f32));
        }
        let features = values.len() / classes;
        Ok(Self {
            name: name.into(),
            classes,
            features,
            values,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn features(&self) -> usize {
        self.features
    }
    /// Row-major fixed codes, before the shared trainable encoder.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

fn normal(random: &mut Random) -> f32 {
    let radius = (-2.0 * (1.0 - f64::from(random.unit())).ln()).sqrt();
    let angle = std::f64::consts::TAU * f64::from(random.unit());
    (radius * angle.cos()) as f32
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod numerical_kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        classes >= 2, features >= 1, width >= 2, code.len() == classes * features,
        bank.len() == features * width, out.len() == classes * width))]
    pub fn hidden<const S: usize>(
        classes: usize,
        features: usize,
        width: usize,
        code: &[f32],
        bank: &[[f32; 3]],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = (S, classes);
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let (x, c) = (item / width, item % width);
            let mut sum = 0.0;
            let mut f = 0;
            while f < features {
                sum += code[x * features + f] * bank[f * width + c][0];
                f += 1;
            }
            *output = sum;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        classes >= 2, pairs >= 1, width == 2 * pairs, nonlinear <= 1,
        hidden.len() == classes * width, mixer.len() == width * width,
        out.len() == 2 * classes * width))]
    pub fn materialize<const S: usize>(
        classes: usize,
        pairs: usize,
        width: usize,
        nonlinear: usize,
        hidden: &[f32],
        mixer: &[[f32; 3]],
        mut out: DisjointSlice<[f32; 3]>,
    ) {
        let _ = (S, pairs);
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let right = item >= classes * width;
            let x = item / width % classes;
            let c = item % width;
            let target = if right { c ^ 1 } else { c };
            let mut value = hidden[x * width + target];
            if nonlinear == 1 {
                value = 0.0;
                let mut j = 0;
                while j < width {
                    value += hidden[x * width + j].max(0.0) * mixer[j * width + target][0];
                    j += 1;
                }
            }
            if right && c.is_multiple_of(2) {
                value = -value;
            }
            *output = [value, 0.0, 0.0];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        classes >= 2, pairs >= 1, width == 2 * pairs,
        gradient.len() == 2 * classes * width, out.len() == classes * width))]
    pub fn embedding_gradient<const S: usize>(
        classes: usize,
        pairs: usize,
        width: usize,
        gradient: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = (S, pairs);
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let (x, c) = (item / width, item % width);
            let other = gradient[(classes + x) * width + (c ^ 1)];
            *output = gradient[item] + if c.is_multiple_of(2) { other } else { -other };
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        classes >= 2, width >= 2, nonlinear <= 1, hidden.len() == classes * width,
        gradient.len() == classes * width, mixer.len() == width * width,
        out.len() == classes * width))]
    pub fn hidden_gradient<const S: usize>(
        classes: usize,
        width: usize,
        nonlinear: usize,
        hidden: &[f32],
        gradient: &[f32],
        mixer: &[[f32; 3]],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = (S, classes);
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let (x, j) = (item / width, item % width);
            let mut value = gradient[item];
            if nonlinear == 1 {
                value = 0.0;
                if hidden[item] > 0.0 {
                    let mut c = 0;
                    while c < width {
                        value += gradient[x * width + c] * mixer[j * width + c][0];
                        c += 1;
                    }
                }
            }
            *output = value;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        classes >= 2, features >= 1, width >= 2, code.len() == classes * features,
        gradient.len() == classes * width, out.len() == features * width))]
    pub fn bank_gradient<const S: usize>(
        classes: usize,
        features: usize,
        width: usize,
        code: &[f32],
        gradient: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let (f, c) = (item / width, item % width);
            let mut sum = 0.0;
            let mut x = 0;
            while x < classes {
                sum += code[x * features + f] * gradient[x * width + c];
                x += 1;
            }
            *output = sum;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        classes >= 2, width >= 2, hidden.len() == classes * width,
        gradient.len() == classes * width, out.len() == width * width))]
    pub fn mixer_gradient<const S: usize>(
        classes: usize,
        width: usize,
        hidden: &[f32],
        gradient: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let (j, c) = (item / width, item % width);
            let mut sum = 0.0;
            let mut x = 0;
            while x < classes {
                sum += hidden[x * width + j].max(0.0) * gradient[x * width + c];
                x += 1;
            }
            *output = sum;
        }
    }
}

/// `e(x) = code(x) P`, optionally `ReLU(code(x) P) Q`, shared by a and b.
/// The core sees `e(a) + Rotate90(e(b))` and predicts 97 independent logits.
pub struct GpuNumericalModel {
    model: GpuSimpleModel,
    code: NumericalCode,
    nonlinear: bool,
    bank: Parameters,
    mixer: Parameters,
    codes: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    embedding_gradient: DeviceBuffer<f32>,
    hidden_gradient: DeviceBuffer<f32>,
    module: numerical_kernels::LoadedModule,
}

impl GpuNumericalModel {
    pub fn new(
        device: usize,
        config: SimpleConfig,
        code: NumericalCode,
        nonlinear: bool,
        seed: u64,
    ) -> Result<Self> {
        if config.architecture != Architecture::ResidualMlp
            || config.encoding != Encoding::OneHot
            || config.width < 2
            || !config.width.is_multiple_of(2)
            || config.modulus != code.classes
        {
            return Err("numerical encoder needs an even-width one-hot residual MLP cache".into());
        }
        let model = GpuSimpleModel::new(device, config, seed)?;
        let stream = &model.kernels.stream;
        let mut random = Random::new(seed ^ 0x6e756d62616e6b01);
        let bank: Vec<_> = (0..code.features * config.width)
            .map(|_| [normal(&mut random) / (config.width as f32).sqrt(), 0.0, 0.0])
            .collect();
        let mut mixer_random = Random::new(seed ^ 0x6e756d6d69786501);
        let mixer: Vec<_> = (0..config.width * config.width)
            .map(|_| {
                [
                    normal(&mut mixer_random) * (2.0 / config.width as f32).sqrt(),
                    0.0,
                    0.0,
                ]
            })
            .collect();
        Ok(Self {
            bank: Parameters {
                state: DeviceBuffer::from_host(stream, &bank)?,
                gradient: DeviceBuffer::zeroed(stream, bank.len())?,
            },
            mixer: Parameters {
                state: DeviceBuffer::from_host(stream, &mixer)?,
                gradient: DeviceBuffer::zeroed(stream, mixer.len())?,
            },
            codes: DeviceBuffer::from_host(stream, &code.values)?,
            hidden: DeviceBuffer::zeroed(stream, config.modulus * config.width)?,
            embedding_gradient: DeviceBuffer::zeroed(stream, config.modulus * config.width)?,
            hidden_gradient: DeviceBuffer::zeroed(stream, config.modulus * config.width)?,
            module: numerical_kernels::load(stream.context())?,
            model,
            code,
            nonlinear,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.model.config.parameter_count() - self.model.parameters[0].state.len()
            + self.bank.state.len()
            + usize::from(self.nonlinear) * self.mixer.state.len()
    }

    fn generate(&mut self) -> Result<()> {
        let c = self.model.config;
        let k = &self.model.kernels;
        let launch = k.launches.get(config::<1>(self.hidden.len())?, |l| {
            self.module.prepare_hidden::<0>(l)
        })?;
        self.module.hidden(
            &k.stream,
            &launch,
            c.modulus,
            self.code.features,
            c.width,
            &self.codes,
            &self.bank.state,
            &mut self.hidden,
        )?;
        let launch = k.launches.get(config::<1>(2 * self.hidden.len())?, |l| {
            self.module.prepare_materialize::<0>(l)
        })?;
        self.module.materialize(
            &k.stream,
            &launch,
            c.modulus,
            c.width / 2,
            c.width,
            usize::from(self.nonlinear),
            &self.hidden,
            &self.mixer.state,
            &mut self.model.parameters[0].state,
        )?;
        Ok(())
    }

    fn reduce_gradient(&mut self) -> Result<()> {
        let c = self.model.config;
        let k = &self.model.kernels;
        let launch = k.launches.get(config::<1>(self.hidden.len())?, |l| {
            self.module.prepare_embedding_gradient::<0>(l)
        })?;
        self.module.embedding_gradient(
            &k.stream,
            &launch,
            c.modulus,
            c.width / 2,
            c.width,
            &self.model.parameters[0].gradient,
            &mut self.embedding_gradient,
        )?;
        let launch = k.launches.get(config::<1>(self.hidden.len())?, |l| {
            self.module.prepare_hidden_gradient::<0>(l)
        })?;
        self.module.hidden_gradient(
            &k.stream,
            &launch,
            c.modulus,
            c.width,
            usize::from(self.nonlinear),
            &self.hidden,
            &self.embedding_gradient,
            &self.mixer.state,
            &mut self.hidden_gradient,
        )?;
        let launch = k.launches.get(config::<1>(self.bank.state.len())?, |l| {
            self.module.prepare_bank_gradient::<0>(l)
        })?;
        self.module.bank_gradient(
            &k.stream,
            &launch,
            c.modulus,
            self.code.features,
            c.width,
            &self.codes,
            &self.hidden_gradient,
            &mut self.bank.gradient,
        )?;
        if self.nonlinear {
            let launch = k.launches.get(config::<1>(self.mixer.state.len())?, |l| {
                self.module.prepare_mixer_gradient::<0>(l)
            })?;
            self.module.mixer_gradient(
                &k.stream,
                &launch,
                c.modulus,
                c.width,
                &self.hidden,
                &self.embedding_gradient,
                &mut self.mixer.gradient,
            )?;
        }
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
        let k = &self.model.kernels;
        k.adam(optimizer, corrections, &mut self.bank)?;
        if self.nonlinear {
            k.adam(optimizer, corrections, &mut self.mixer)?;
        }
        for group in self.model.parameters.iter_mut().skip(1) {
            k.adam(optimizer, corrections, group)?;
        }
        Ok(())
    }

    pub fn parameters(&mut self) -> Result<Vec<f32>> {
        let mut result = Vec::with_capacity(self.parameter_count());
        for group in std::iter::once(&self.bank).chain(self.nonlinear.then_some(&self.mixer)) {
            let mut values = DeviceBuffer::zeroed(&self.model.kernels.stream, group.state.len())?;
            self.model.kernels.weights(&group.state, &mut values)?;
            result.extend(values.to_host_vec(&self.model.kernels.stream)?);
        }
        let full = self.model.parameters()?;
        result.extend_from_slice(&full[self.model.parameters[0].state.len()..]);
        validate_finite(&result, "numerical encoder parameters")?;
        Ok(result)
    }

    /// Compare all code/encoder/core/readout derivatives to the independent
    /// f64 model, with a separate CPU chain rule through the shared encoder.
    pub fn verify(device: usize) -> Result<()> {
        use crate::{
            gpt::{EquationDataset, ModularOperation},
            simple::ReferenceSimpleModel,
        };
        let c = SimpleConfig {
            architecture: Architecture::ResidualMlp,
            modulus: 97,
            width: 6,
            expansion: 1,
            ..SimpleConfig::default()
        };
        let data = EquationDataset::new(97, ModularOperation::Divide, 0.5, 11)?;
        let equations = &data.split(false)[..5];
        let (features, targets) = c.encode(equations)?;
        for name in NumericalCode::NAMES {
            for nonlinear in [false, true] {
                let code = NumericalCode::new(97, name, if name == "random-bits" { 8 } else { 3 })?;
                let mut gpu = Self::new(device, c, code.clone(), nonlinear, 11)?;
                let parameters = gpu.parameters()?;
                let bank_end = code.features * c.width;
                let prefix = bank_end + usize::from(nonlinear) * c.width * c.width;
                let mut hidden = vec![0.0_f64; c.modulus * c.width];
                let mut encoded = hidden.clone();
                for x in 0..c.modulus {
                    for j in 0..c.width {
                        hidden[x * c.width + j] = (0..code.features)
                            .map(|f| {
                                f64::from(code.values[x * code.features + f])
                                    * f64::from(parameters[f * c.width + j])
                            })
                            .sum();
                    }
                    for j in 0..c.width {
                        encoded[x * c.width + j] = if nonlinear {
                            (0..c.width)
                                .map(|i| {
                                    hidden[x * c.width + i].max(0.0)
                                        * f64::from(parameters[bank_end + i * c.width + j])
                                })
                                .sum()
                        } else {
                            hidden[x * c.width + j]
                        };
                    }
                }
                let mut full = Vec::new();
                full.extend(encoded.iter().map(|&v| v as f32));
                for x in 0..c.modulus {
                    for j in 0..c.width {
                        full.push(
                            (encoded[x * c.width + (j ^ 1)]
                                * if j.is_multiple_of(2) { -1.0 } else { 1.0 })
                                as f32,
                        );
                    }
                }
                full.extend_from_slice(&parameters[prefix..]);
                let reference = ReferenceSimpleModel::from_parameters(c, full)?;
                let selected = [2_usize, 0, 3, 2];
                let x: Vec<_> = selected
                    .iter()
                    .flat_map(|&i| {
                        features[i * c.input_features()..(i + 1) * c.input_features()]
                            .iter()
                            .copied()
                    })
                    .collect();
                let y: Vec<_> = selected.iter().map(|&i| targets[i]).collect();
                let (loss, gradient) = reference.loss_and_gradient(&x, &y)?;
                let mut expected = vec![0.0_f64; parameters.len()];
                for x in 0..c.modulus {
                    let mut de = vec![0.0; c.width];
                    for j in 0..c.width {
                        de[j] = gradient[x * c.width + j]
                            + gradient[(c.modulus + x) * c.width + (j ^ 1)]
                                * if j.is_multiple_of(2) { 1.0 } else { -1.0 };
                    }
                    for j in 0..c.width {
                        let dh = if nonlinear {
                            for k in 0..c.width {
                                expected[bank_end + j * c.width + k] +=
                                    hidden[x * c.width + j].max(0.0) * de[k];
                            }
                            if hidden[x * c.width + j] > 0.0 {
                                (0..c.width)
                                    .map(|k| {
                                        de[k] * f64::from(parameters[bank_end + j * c.width + k])
                                    })
                                    .sum()
                            } else {
                                0.0
                            }
                        } else {
                            de[j]
                        };
                        for f in 0..code.features {
                            expected[f * c.width + j] +=
                                f64::from(code.values[x * code.features + f]) * dh;
                        }
                    }
                }
                expected[prefix..].copy_from_slice(&gradient[2 * c.modulus * c.width..]);
                gpu.generate()?;
                let table = gpu.upload_table(&features, &targets)?;
                let indices = [4, 2, 0, 3, 2];
                let (actual_loss, full_gradient) =
                    gpu.model.loss_and_gradient(&table, &indices[1..])?;
                gpu.reduce_gradient()?;
                let mut actual: Vec<_> = gpu
                    .bank
                    .gradient
                    .to_host_vec(&gpu.model.kernels.stream)?
                    .into_iter()
                    .map(f64::from)
                    .collect();
                if nonlinear {
                    actual.extend(
                        gpu.mixer
                            .gradient
                            .to_host_vec(&gpu.model.kernels.stream)?
                            .into_iter()
                            .map(f64::from),
                    );
                }
                actual.extend_from_slice(&full_gradient[2 * c.modulus * c.width..]);
                if (loss - actual_loss).abs() > 1e-5 {
                    return Err(format!("{name} loss mismatch").into());
                }
                for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                    if !a.is_finite() || (a - b).abs() > 2e-4 * (1.0 + b.abs()) {
                        return Err(format!(
                            "{name} nonlinear={nonlinear} gradient {i}: GPU {a}, f64 {b}"
                        )
                        .into());
                    }
                }
                let optimizer = AdamWOptions {
                    learning_rate: 0.001,
                    weight_decay: 0.01,
                    ..AdamWOptions::default()
                };
                let batches = gpu.upload_batches(&table, &indices)?;
                gpu.train_batch(&table, &batches, 1, 4, optimizer)?;
                for (i, ((&p, &g), a)) in parameters
                    .iter()
                    .zip(&actual)
                    .zip(gpu.parameters()?)
                    .enumerate()
                {
                    let g = g as f32;
                    let m = (1.0 - optimizer.beta1) * g;
                    let v = (1.0 - optimizer.beta2) * g * g;
                    let wanted = p - optimizer.learning_rate
                        * (m / (1.0 - optimizer.beta1)
                            / ((v / (1.0 - optimizer.beta2)).sqrt() + optimizer.epsilon)
                            + optimizer.weight_decay * p);
                    if (a - wanted).abs() > 4e-6 {
                        return Err(format!("{name} AdamW mismatch at {i}").into());
                    }
                }
                println!(
                    "PASS numerical {name} nonlinear={nonlinear}: f64 loss/all gradients, independent readout, shared encoder, repeated/offset batch, AdamW"
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_codes_preserve_all_identities_and_control_norm() {
        for name in NumericalCode::NAMES {
            let code = NumericalCode::new(97, name, 8).unwrap();
            for (x, row) in code.values.chunks_exact(code.features).enumerate() {
                assert!(row.iter().all(|v| v.is_finite()));
                let norm2 = row.iter().map(|v| v * v).sum::<f32>();
                if name == "integer-bits" {
                    assert!(norm2 > 0.0 && norm2 <= 1.0 + 1e-6);
                } else {
                    assert!((norm2 - 1.0).abs() < 1e-6);
                }
                for other in code.values[..x * code.features].chunks_exact(code.features) {
                    let distance: f32 = row.iter().zip(other).map(|(a, b)| (a - b).powi(2)).sum();
                    assert!(distance > 1e-5, "collision for {name}, x={x}");
                }
            }
        }
    }

    #[test]
    fn random_binary_codes_are_fixed_binary_and_collision_free() {
        let code = NumericalCode::new(97, "random-bits", 8).unwrap();
        assert_eq!(code.features(), 16);
        assert!(code.values().iter().all(|&v| v == -0.25 || v == 0.25));
        assert_eq!(
            code.values(),
            NumericalCode::new(97, "random-bits", 8).unwrap().values()
        );
        assert!(NumericalCode::new(97, "random-bits", 3).is_err());
        // Cover a complete code space, forcing duplicate draws to be rejected.
        let full = NumericalCode::new(16, "random-bits", 2).unwrap();
        let words: BTreeSet<_> = full
            .values()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|row| {
                row.iter().enumerate().fold(0_u32, |word, (bit, &v)| {
                    assert!(v == -0.5 || v == 0.5);
                    word | (u32::from(v > 0.0) << bit)
                })
            })
            .collect();
        assert_eq!(words, (0..16).collect());
    }

    #[test]
    fn integer_bit_codes_decode_to_values_and_have_one_shared_bias() {
        let code = NumericalCode::new(97, "integer-bits", 8).unwrap();
        assert_eq!(code.features(), 8);
        let scale = 8.0_f32.sqrt().recip();
        for (value, row) in code.values().as_chunks::<8>().0.iter().enumerate() {
            assert!((row[7] - scale).abs() < 1e-7);
            let mut decoded = 0;
            for &bit in &row[..7] {
                assert!(bit == 0.0 || (bit - scale).abs() < 1e-7);
                decoded = 2 * decoded + usize::from(bit > 0.0);
            }
            assert_eq!(decoded, value);
        }
        assert!(code.values()[..7].iter().all(|&v| v == 0.0));
    }
}
