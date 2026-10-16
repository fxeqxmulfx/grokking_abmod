//! Native CUDA experiments with two scalar features, one linear output and MSE.
use grokking_abmod::{
    Result,
    gpt::{AdamWOptions, Equation, EquationDataset, EquationSampler, ModularOperation},
    simple::{
        GpuRegressionMlp, RegressionBiasInitialization as BiasInitialization, RegressionConfig,
        RegressionTable,
    },
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    str::FromStr,
    time::Instant,
};

struct Options {
    config: RegressionConfig,
    operation: ModularOperation,
    modulus: usize,
    train_limit: Option<usize>,
    scale: Scale,
    bias_initialization: BiasInitialization,
    bias_scale: f32,
    cosine_schedule: bool,
    min_lr_ratio: f32,
    epochs: usize,
    batch: usize,
    report_every: usize,
    optimizer: AdamWOptions,
    seed: u64,
    device: usize,
    output: PathBuf,
    verify: bool,
}

#[derive(Clone, Copy)]
enum Scale {
    Raw,
    Unit,
    Centered,
}

impl Scale {
    fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Unit => "unit",
            Self::Centered => "centered",
        }
    }

    fn transform(self, modulus: usize) -> (f64, f64) {
        match self {
            Self::Raw => (1.0, 0.0),
            Self::Unit => ((modulus - 1) as f64, 0.0),
            Self::Centered => ((modulus - 1) as f64, 0.5),
        }
    }
}

fn learning_rate_factor(step: usize, total: usize, cosine: bool, minimum: f32) -> f32 {
    let warmup = step.min(10) as f32 / 10.0;
    if !cosine || total <= 10 || step <= 10 {
        return warmup;
    }
    let progress = (step - 10) as f64 / (total - 10) as f64;
    warmup
        * (minimum
            + (1.0 - minimum) * (0.5 * (1.0 + (std::f64::consts::PI * progress).cos())) as f32)
}

fn value<T: FromStr>(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<T> {
    arguments
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))?
        .parse()
        .map_err(|_| format!("invalid value for {flag}").into())
}

impl Options {
    fn parse() -> Result<Self> {
        let mut options = Self {
            config: RegressionConfig {
                width: 64,
                depth: 2,
                pre_rmsnorm: false,
                residual: false,
            },
            operation: ModularOperation::Divide,
            modulus: 97,
            train_limit: None,
            scale: Scale::Unit,
            bias_initialization: BiasInitialization::Zero,
            bias_scale: 1.0,
            cosine_schedule: false,
            min_lr_ratio: 0.001,
            epochs: 1000,
            batch: 512,
            report_every: 10,
            optimizer: AdamWOptions {
                learning_rate: 0.001,
                weight_decay: 0.0,
                beta2: 0.98,
                ..AdamWOptions::default()
            },
            seed: 11,
            device: 0,
            output: PathBuf::from("target/scalar-mlp"),
            verify: false,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(flag) = arguments.next() {
            match flag.as_str() {
                "--width" => options.config.width = value(&mut arguments, &flag)?,
                "--depth" => options.config.depth = value(&mut arguments, &flag)?,
                "--pre-rmsnorm" => options.config.pre_rmsnorm = true,
                "--residual" => options.config.residual = true,
                "--operation" => options.operation = value(&mut arguments, &flag)?,
                "--modulus" => options.modulus = value(&mut arguments, &flag)?,
                "--train-limit" => options.train_limit = Some(value(&mut arguments, &flag)?),
                "--scale" => {
                    let scale: String = value(&mut arguments, &flag)?;
                    options.scale = match scale.as_str() {
                        "unit" => Scale::Unit,
                        "raw" => Scale::Raw,
                        "centered" => Scale::Centered,
                        _ => return Err("scale must be unit, raw or centered".into()),
                    };
                }
                "--bias-init" => {
                    let mode: String = value(&mut arguments, &flag)?;
                    options.bias_initialization = match mode.as_str() {
                        "zero" => BiasInitialization::Zero,
                        "uniform" => BiasInitialization::Uniform,
                        "domain" => BiasInitialization::Domain,
                        _ => {
                            return Err(
                                "bias initialization must be zero, uniform or domain".into()
                            );
                        }
                    };
                }
                "--bias-scale" => options.bias_scale = value(&mut arguments, &flag)?,
                "--schedule" => {
                    let schedule: String = value(&mut arguments, &flag)?;
                    options.cosine_schedule = match schedule.as_str() {
                        "constant" => false,
                        "cosine" => true,
                        _ => return Err("schedule must be constant or cosine".into()),
                    };
                }
                "--min-lr-ratio" => options.min_lr_ratio = value(&mut arguments, &flag)?,
                "--epochs" => options.epochs = value(&mut arguments, &flag)?,
                "--batch" => options.batch = value(&mut arguments, &flag)?,
                "--report-every" => options.report_every = value(&mut arguments, &flag)?,
                "--learning-rate" => {
                    options.optimizer.learning_rate = value(&mut arguments, &flag)?
                }
                "--weight-decay" => options.optimizer.weight_decay = value(&mut arguments, &flag)?,
                "--beta2" => options.optimizer.beta2 = value(&mut arguments, &flag)?,
                "--seed" => options.seed = value(&mut arguments, &flag)?,
                "--device" => options.device = value(&mut arguments, &flag)?,
                "--output" => options.output = value(&mut arguments, &flag)?,
                "--verify" => options.verify = true,
                "--help" | "-h" => {
                    println!(
                        "scalar-mlp [--width N] [--depth N] [--pre-rmsnorm] [--residual] [--operation div|mul] [--modulus P] [--train-limit N] [--scale unit|raw|centered] [--bias-init zero|uniform|domain] [--bias-scale X] [--schedule constant|cosine] [--min-lr-ratio X] [--epochs N] [--batch N] [--report-every N] [--learning-rate X] [--weight-decay X] [--beta2 X] [--seed N] [--device N] [--output PATH] [--verify]\nDepth counts all ReLU hidden layers. RMSNorm and residual apply only after the first hidden layer. Output is linear; loss is mean squared error. Centered scale subtracts 0.5 from unit-scaled operands and target. Train limits take a seeded prefix of the original training half; held-out equations stay unchanged. Bias modes preserve Kaiming matrices: uniform uses +/-scale/sqrt(fan_in); domain anchors first-layer hinges inside the operand box, with uniform biases in later layers."
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown scalar-mlp option: {flag}").into()),
            }
        }
        options.config.validate()?;
        options.optimizer.validate()?;
        if options.epochs == 0 || options.batch == 0 || options.report_every == 0 {
            return Err("epochs, batch and report interval must be positive".into());
        }
        if !options.min_lr_ratio.is_finite() || !(0.0..=1.0).contains(&options.min_lr_ratio) {
            return Err("minimum learning-rate ratio must be in 0..1".into());
        }
        if !options.bias_scale.is_finite() || options.bias_scale < 0.0 {
            return Err("bias scale must be finite and nonnegative".into());
        }
        if !matches!(
            options.operation,
            ModularOperation::Divide | ModularOperation::Multiply
        ) {
            return Err("this experiment supports only div and mul".into());
        }
        Ok(options)
    }
}

struct Split {
    table: RegressionTable,
    targets: Vec<f32>,
    answers: Vec<usize>,
}

fn upload(
    model: &GpuRegressionMlp,
    equations: &[Equation],
    divisor: f32,
    offset: f32,
) -> Result<Split> {
    let features: Vec<_> = equations
        .iter()
        .flat_map(|equation| {
            [
                equation.a as f32 / divisor - offset,
                equation.b as f32 / divisor - offset,
            ]
        })
        .collect();
    let targets: Vec<_> = equations
        .iter()
        .map(|equation| equation.answer as f32 / divisor - offset)
        .collect();
    let answers = equations.iter().map(|equation| equation.answer).collect();
    Ok(Split {
        table: model.upload_table(&features, &targets)?,
        targets,
        answers,
    })
}

#[derive(Clone, Copy)]
struct Metrics {
    mse: f64,
    rmse: f64,
    accuracy: f64,
}

fn metrics(predictions: &[f32], split: &Split, scale: f64, offset: f64) -> Result<Metrics> {
    if predictions.len() != split.answers.len()
        || predictions.iter().any(|value| !value.is_finite())
    {
        return Err("nonfinite or incomplete scalar predictions".into());
    }
    let mut mse = 0.0;
    let mut raw_mse = 0.0;
    let mut correct = 0;
    for ((&prediction, &target), &answer) in
        predictions.iter().zip(&split.targets).zip(&split.answers)
    {
        mse += (f64::from(prediction) - f64::from(target)).powi(2);
        let raw = (f64::from(prediction) + offset) * scale;
        raw_mse += (raw - answer as f64).powi(2);
        correct += usize::from(raw.round() == answer as f64);
    }
    let rows = predictions.len() as f64;
    Ok(Metrics {
        mse: mse / rows,
        rmse: (raw_mse / rows).sqrt(),
        accuracy: correct as f64 / rows,
    })
}

fn checkpoint(path: PathBuf, parameters: &[f32]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for parameter in parameters {
        writer.write_all(&parameter.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn run(options: Options) -> Result<()> {
    if options.verify {
        return GpuRegressionMlp::verify(options.device);
    }
    let dataset = EquationDataset::new(options.modulus, options.operation, 0.5, options.seed)?;
    let training_pool = dataset.split(false);
    let train_size = options.train_limit.unwrap_or(training_pool.len());
    if !(2..=training_pool.len()).contains(&train_size) {
        return Err(
            "train limit must be at least two and at most the original training half".into(),
        );
    }
    if options.batch > train_size {
        return Err("batch exceeds training split".into());
    }
    let (scale, output_offset) = options.scale.transform(options.modulus);
    let maximum = ((options.modulus - 1) as f64 / scale - output_offset) as f32;
    let min_b = (f64::from(u8::from(options.operation == ModularOperation::Divide)) / scale
        - output_offset) as f32;
    let initial = options.config.initial_parameters_with_biases(
        options.seed,
        options.bias_initialization,
        options.bias_scale,
        [[-output_offset as f32, maximum], [min_b, maximum]],
    )?;
    let mut model = GpuRegressionMlp::from_parameters(options.device, options.config, &initial)?;
    let train = upload(
        &model,
        &training_pool[..train_size],
        scale as f32,
        output_offset as f32,
    )?;
    let test = upload(
        &model,
        dataset.split(true),
        scale as f32,
        output_offset as f32,
    )?;
    std::fs::create_dir_all(&options.output)?;
    checkpoint(
        options.output.join("initial-model.bin"),
        &model.parameters()?,
    )?;
    let mean = train
        .answers
        .iter()
        .map(|&answer| answer as f64)
        .sum::<f64>()
        / train_size as f64;
    let constant_test_rmse = (test
        .answers
        .iter()
        .map(|&answer| (mean - answer as f64).powi(2))
        .sum::<f64>()
        / test.answers.len() as f64)
        .sqrt();
    let constant_test_accuracy = test
        .answers
        .iter()
        .filter(|&&answer| mean.round() == answer as f64)
        .count() as f64
        / test.answers.len() as f64;
    let header = format!(
        "architecture=scalar-mlp\ninputs=2\noutputs=1\nactivation=relu\nloss=mse_mean\noperation={}\nmodulus={}\ntrain_fraction=0.5\ntraining_equations={}\nvalidation_equations={}\nseed={}\nwidth={}\ndepth={}\npre_rmsnorm={}\nresidual={}\nfinal_rmsnorm=false\nrmsnorm_epsilon=1e-5\nnormalization_affine=false\nbias=true\ninitialization=kaiming_normal_hidden_linear_gain_output\nbias_initialization={}\nbias_initialization_scale={}\nscale={}\noutput_scale={scale}\noutput_offset={output_offset}\nparameters={}\nlearning_rate={}\nweight_decay={}\nweight_decay_bias=true\nbetas=0.9,{}\nepsilon=1e-8\nwarmup_updates=10\nschedule={}\nminimum_lr_ratio={}\ntraining_pool_equations={}\nunused_training_equations={}\nbatch={}\nrequested_epochs={}\nreport_every_epochs={}\nevaluation=complete_splits\nepoch_definition=one_visit_to_every_training_equation\nepoch_tail=short_batch\nrounding=nearest_integer_no_clip_no_modulo\nconstant_train_mean={mean}\nconstant_test_rmse_residue={constant_test_rmse}\nconstant_test_accuracy={constant_test_accuracy}\n",
        options.operation,
        options.modulus,
        train_size,
        test.answers.len(),
        options.seed,
        options.config.width,
        options.config.depth,
        options.config.pre_rmsnorm,
        options.config.residual,
        options.bias_initialization.name(),
        options.bias_scale,
        options.scale.name(),
        options.config.parameter_count(),
        options.optimizer.learning_rate,
        options.optimizer.weight_decay,
        options.optimizer.beta2,
        if options.cosine_schedule {
            "cosine"
        } else {
            "constant"
        },
        options.min_lr_ratio,
        training_pool.len(),
        training_pool.len() - train_size,
        options.batch,
        options.epochs,
        options.report_every
    );
    std::fs::write(
        options.output.join("run.txt"),
        format!("{header}status=running\n"),
    )?;
    let mut writer = BufWriter::new(File::create(options.output.join("loss.csv"))?);
    writeln!(
        writer,
        "epoch,step,train_mse,test_mse,train_rmse_residue,test_rmse_residue,train_accuracy,test_accuracy,elapsed_seconds"
    )?;
    let mut sampler = EquationSampler::new(train_size, options.seed)?;
    let mut indices: Vec<_> = (0..train_size as u32).collect();
    let mut batches = model.upload_batches(&train.table, &indices)?;
    let mut step = 0;
    let mut completed_epochs = 0;
    let total_steps = options
        .epochs
        .checked_mul(train_size.div_ceil(options.batch))
        .ok_or("update count overflow")?;
    let start = Instant::now();
    let mut status = "complete";
    for epoch in 0..=options.epochs {
        if epoch > 0 {
            indices.clear();
            while indices.len() < train_size {
                indices.extend(
                    sampler
                        .sample(options.batch)?
                        .iter()
                        .map(|&index| index as u32),
                );
            }
            model.update_batches(&train.table, &mut batches, &indices)?;
            for offset in (0..train_size).step_by(options.batch) {
                step += 1;
                let optimizer = AdamWOptions {
                    learning_rate: options.optimizer.learning_rate
                        * learning_rate_factor(
                            step,
                            total_steps,
                            options.cosine_schedule,
                            options.min_lr_ratio,
                        ),
                    ..options.optimizer
                };
                model.train_batch(
                    &train.table,
                    &batches,
                    offset,
                    options.batch.min(train_size - offset),
                    optimizer,
                )?;
            }
        }
        completed_epochs = epoch;
        if epoch % options.report_every != 0 && epoch != options.epochs {
            continue;
        }
        let result = (|| -> Result<(Metrics, Metrics)> {
            Ok((
                metrics(
                    &model.predictions(&train.table)?,
                    &train,
                    scale,
                    output_offset,
                )?,
                metrics(
                    &model.predictions(&test.table)?,
                    &test,
                    scale,
                    output_offset,
                )?,
            ))
        })();
        let (train_metrics, test_metrics) = match result {
            Ok(metrics) => metrics,
            Err(error) if error.to_string().contains("finite") => {
                status = "nonfinite";
                break;
            }
            Err(error) => return Err(error),
        };
        writeln!(
            writer,
            "{epoch},{step},{:.12},{:.12},{:.9},{:.9},{:.9},{:.9},{:.6}",
            train_metrics.mse,
            test_metrics.mse,
            train_metrics.rmse,
            test_metrics.rmse,
            train_metrics.accuracy,
            test_metrics.accuracy,
            start.elapsed().as_secs_f64()
        )?;
        writer.flush()?;
    }
    if status == "complete" {
        checkpoint(options.output.join("model.bin"), &model.parameters()?)?;
        let predictions = model.predictions(&test.table)?;
        let mut predictions_writer =
            BufWriter::new(File::create(options.output.join("predictions.csv"))?);
        writeln!(predictions_writer, "a,b,target,prediction_residue")?;
        for (equation, prediction) in dataset.split(true).iter().zip(predictions) {
            writeln!(
                predictions_writer,
                "{},{},{},{:.9}",
                equation.a,
                equation.b,
                equation.answer,
                (f64::from(prediction) + output_offset) * scale
            )?;
        }
    }
    std::fs::write(
        options.output.join("run.txt"),
        format!(
            "{header}status={status}\ncompleted_epochs={completed_epochs}\nupdates={step}\nelapsed_seconds={:.6}\n",
            start.elapsed().as_secs_f64()
        ),
    )?;
    println!(
        "scalar-mlp width={} depth={} pre_rmsnorm={} residual={} operation={} scale={} seed={} status={status} epochs={completed_epochs} seconds={:.3}",
        options.config.width,
        options.config.depth,
        options.config.pre_rmsnorm,
        options.config.residual,
        options.operation,
        options.scale.name(),
        options.seed,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() {
    if let Err(error) = Options::parse().and_then(run) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_initialization_preserves_matrices_and_places_first_hinges_in_domain() {
        for (width, depth) in [(16, 1), (128, 8)] {
            let config = RegressionConfig {
                width,
                depth,
                pre_rmsnorm: false,
                residual: false,
            };
            let expected = config.initial_parameters(11);
            for bounds in [
                [[0.0, 1.0], [0.0, 1.0]],
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[0.0, 96.0], [1.0, 96.0]],
            ] {
                let uniform = config
                    .initial_parameters_with_biases(11, BiasInitialization::Uniform, 1.0, bounds)
                    .unwrap();
                for mode in [
                    BiasInitialization::Zero,
                    BiasInitialization::Uniform,
                    BiasInitialization::Domain,
                ] {
                    let actual = config
                        .initial_parameters_with_biases(11, mode, 1.0, bounds)
                        .unwrap();
                    assert_eq!(
                        actual,
                        config
                            .initial_parameters_with_biases(11, mode, 1.0, bounds)
                            .unwrap()
                    );
                    let mut offset = 0;
                    for (index, layer) in config.layers().iter().enumerate() {
                        let end = offset + layer.parameters();
                        assert_eq!(actual[offset..end], expected[offset..end]);
                        if mode == BiasInitialization::Zero {
                            assert_eq!(
                                actual[end..end + layer.outputs],
                                expected[end..end + layer.outputs]
                            );
                        } else if index > 0 {
                            assert_eq!(
                                actual[end..end + layer.outputs],
                                uniform[end..end + layer.outputs]
                            );
                        }
                        offset = end + layer.outputs;
                    }
                    if mode == BiasInitialization::Domain {
                        for channel in 0..width {
                            let a = actual[channel];
                            let b = actual[width + channel];
                            let bias = actual[2 * width + channel];
                            let corners: Vec<_> = bounds[0]
                                .iter()
                                .flat_map(|&x| bounds[1].iter().map(move |&y| a * x + b * y + bias))
                                .collect();
                            assert!(
                                corners.iter().copied().fold(f32::NEG_INFINITY, f32::max) > 0.0
                            );
                            assert!(corners.iter().copied().fold(f32::INFINITY, f32::min) < 0.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cosine_schedule_preserves_warmup_and_finishes_at_requested_floor() {
        for total in [11, 100, 100_000] {
            for step in 1..=10 {
                assert_eq!(
                    learning_rate_factor(step, total, true, 0.001),
                    step as f32 / 10.0
                );
            }
            let mut previous = 1.0;
            for step in 11..=total {
                let next = learning_rate_factor(step, total, true, 0.001);
                assert!(next <= previous && next >= 0.001);
                previous = next;
            }
            assert!((previous - 0.001).abs() < 1e-7);
        }
        for step in 1..=100 {
            assert_eq!(
                learning_rate_factor(step, 100, false, 0.001),
                step.min(10) as f32 / 10.0
            );
        }
    }

    #[test]
    fn numeric_scales_preserve_every_integer_answer_when_restored() {
        for modulus in [2, 7, 17, 97] {
            for scale in [Scale::Raw, Scale::Unit, Scale::Centered] {
                let (divisor, offset) = scale.transform(modulus);
                for residue in 0..modulus {
                    let target = residue as f32 / divisor as f32 - offset as f32;
                    let restored = (f64::from(target) + offset) * divisor;
                    assert_eq!(restored.round(), residue as f64);
                    assert!((restored - residue as f64).abs() < 1e-5);
                }
            }
        }
    }
}
