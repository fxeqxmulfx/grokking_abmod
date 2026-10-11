use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use grokking_abmod::{
    Result,
    gpt::{AdamWOptions, EquationDataset, EquationSampler, ModularOperation},
    simple::{Architecture, DeviceBatches, DeviceTable, GpuSimpleModel, SimpleConfig},
};

use super::{
    grokking::{
        Thresholds, epoch_steps, observed, reached_stop, row_metrics, validate_stop_accuracy,
        warmed_optimizer, write_dataset,
    },
    value,
};

struct Options {
    config: SimpleConfig,
    optimizer: AdamWOptions,
    operation: ModularOperation,
    train_fraction: f64,
    steps: usize,
    epochs: Option<usize>,
    batch: Option<usize>,
    report_every: usize,
    stop_validation_accuracy: Option<f64>,
    seed: u64,
    init_seed: Option<u64>,
    device: usize,
    output: PathBuf,
    verify: bool,
    help: bool,
}

impl Options {
    fn parse(mut arguments: impl Iterator<Item = String>) -> Result<Self> {
        let mut options = Self {
            config: SimpleConfig::default(),
            optimizer: AdamWOptions {
                learning_rate: 0.001,
                weight_decay: 1.0,
                beta2: 0.98,
                ..AdamWOptions::default()
            },
            operation: ModularOperation::Add,
            train_fraction: 0.5,
            steps: 100_000,
            epochs: None,
            batch: None,
            report_every: 1000,
            stop_validation_accuracy: None,
            seed: 7,
            init_seed: None,
            device: 0,
            output: PathBuf::from("target/mesh-gpt-simple"),
            verify: false,
            help: false,
        };
        let mut explicit_steps = false;
        while let Some(flag) = arguments.next() {
            match flag.as_str() {
                "--model" => options.config.architecture = value(&mut arguments, &flag)?,
                "--activation" => options.config.activation = value(&mut arguments, &flag)?,
                "--encoding" => options.config.encoding = value(&mut arguments, &flag)?,
                "--width" => options.config.width = value(&mut arguments, &flag)?,
                "--depth" => options.config.depth = value(&mut arguments, &flag)?,
                "--expansion" => options.config.expansion = value(&mut arguments, &flag)?,
                "--connectivity" => options.config.connectivity = value(&mut arguments, &flag)?,
                "--tied-mesh" => options.config.tied_mesh = true,
                "--decay-factors" => options.config.decay_factors = value(&mut arguments, &flag)?,
                "--modulus" => options.config.modulus = value(&mut arguments, &flag)?,
                "--operation" => options.operation = value(&mut arguments, &flag)?,
                "--train-fraction" => options.train_fraction = value(&mut arguments, &flag)?,
                "--steps" => {
                    options.steps = value(&mut arguments, &flag)?;
                    explicit_steps = true;
                }
                "--epochs" => options.epochs = Some(value(&mut arguments, &flag)?),
                "--batch" => options.batch = Some(value(&mut arguments, &flag)?),
                "--report-every" => options.report_every = value(&mut arguments, &flag)?,
                "--stop-validation-accuracy" => {
                    options.stop_validation_accuracy = Some(value(&mut arguments, &flag)?)
                }
                "--learning-rate" => {
                    options.optimizer.learning_rate = value(&mut arguments, &flag)?
                }
                "--weight-decay" => options.optimizer.weight_decay = value(&mut arguments, &flag)?,
                "--seed" => options.seed = value(&mut arguments, &flag)?,
                "--init-seed" => options.init_seed = Some(value(&mut arguments, &flag)?),
                "--device" => options.device = value(&mut arguments, &flag)?,
                "--output" => options.output = value(&mut arguments, &flag)?,
                "--verify" => options.verify = true,
                "--help" | "-h" => options.help = true,
                _ => {
                    return Err(format!(
                        "unknown grok-simple option: {flag}; use grok-simple --help"
                    )
                    .into());
                }
            }
        }
        if explicit_steps && options.epochs.is_some() {
            return Err("choose either --epochs or --steps".into());
        }
        Ok(options)
    }
}

pub fn run(arguments: impl Iterator<Item = String>) -> Result<()> {
    let options = Options::parse(arguments)?;
    if options.help {
        println!(
            "Attention-free baselines on the Mesh GPT grokking protocol\n\
            Usage: cargo oxide run --bin mesh-gpt -- grok-simple [options]\n\n\
            --model NAME          mlp (default), mesh-mlp, residual-mlp, or bilinear\n\
            --encoding NAME       Per operand: one-hot (default), binary (+-1 bits),\n\
                                  scalar ((v-48)/48 for p=97), or discrete-log\n\
                                  ([v==0, centered log to a primitive root]; prime p)\n\
            --activation NAME     relu (default) or square; bilinear has none\n\
            --width N             MLP hidden width, residual stream, or bilinear rank (default: 128)\n\
            --depth N             MLP hidden layers or residual blocks, 1..64 (default: 1);\n\
                                  a mesh MLP's layers after the first are mesh layers\n\
            --expansion N         Residual block hidden width / stream width (default: 4)\n\
            --connectivity NAME   Mesh MLP second parent: neighbor, c+1 (default),\n\
                                  butterfly, c+2^(layer mod floor(log2 width)), or\n\
                                  own, none: each mesh unit scales its own channel\n\
            --tied-mesh           Mesh MLP: every mesh layer uses one stored layer's weights\n\
            --decay-factors N     One-layer MLP: decay channel c's input column and readout\n\
                                  row at decay*s_c^(2/N-1), s_c=(|E_c|^2+|R_c|^2)/2, the\n\
                                  balanced L2 penalty of N factors (default: 2, plain AdamW)\n\
            --operation NAME      As for grok (default: add)\n\
            --modulus P           Prime, 2..254 (default: 97)\n\
            --train-fraction X    Fixed training fraction, 0<X<1 (default: 0.5)\n\
            --steps N             Gradient updates (default: 100000)\n\
            --epochs N            Complete passes over all training equations; replaces --steps\n\
            --stop-validation-accuracy X  End at the first report with validation accuracy\n\
                                  >=X, 0<X<=1 (default: none)\n\
            --batch N             Equations per minibatch (default: min(512, train_size/2))\n\
            --report-every N      Evaluate both complete splits (default: 1000)\n\
            --learning-rate X     Maximum AdamW rate (default: 0.001; 10-step warmup)\n\
            --weight-decay X      AdamW decay (default: 1; betas fixed at 0.9, 0.98)\n\
            --seed N              Split and sampling seed (default: 7)\n\
            --init-seed N         Weight initialization seed (default: --seed)\n\
            --device N            CUDA device ordinal (default: 0)\n\
            --output PATH         Artifacts (default: target/mesh-gpt-simple)\n\
            --verify              Check the CUDA kernels against f64 differentiation and exit"
        );
        return Ok(());
    }
    if options.verify {
        return GpuSimpleModel::verify(options.device);
    }
    train(options)
}

/// One encoded split on the device, with its answers for host-side metrics.
struct Split {
    table: DeviceTable,
    targets: Vec<u32>,
}

/// Up to one epoch of minibatch indices, sampled ahead and uploaded at once.
struct Epoch {
    batches: DeviceBatches,
    minibatches: Vec<[usize; 2]>,
    next: usize,
}

impl Epoch {
    fn sample(
        model: &GpuSimpleModel,
        train: &Split,
        sampler: &mut EquationSampler,
        batch: usize,
        steps: usize,
    ) -> Result<Self> {
        let mut indices = Vec::new();
        let mut minibatches = Vec::with_capacity(steps);
        for _ in 0..steps {
            let sample = sampler.sample(batch)?;
            minibatches.push([indices.len(), sample.len()]);
            indices.extend(sample.iter().map(|&index| index as u32));
        }
        Ok(Self {
            batches: model.upload_batches(&train.table, &indices)?,
            minibatches,
            next: 0,
        })
    }
}

fn train(mut options: Options) -> Result<()> {
    if options.steps == 0 || options.report_every == 0 {
        return Err("steps and report interval must be positive".into());
    }
    validate_stop_accuracy(options.stop_validation_accuracy)?;
    options.optimizer.validate()?;
    let config = options.config;
    config.validate()?;
    let dataset = EquationDataset::new(
        config.modulus,
        options.operation,
        options.train_fraction,
        options.seed,
    )?;
    let training_equations = dataset.split(false).len();
    let batch = options
        .batch
        .unwrap_or_else(|| 512.min(training_equations / 2));
    // Validate the minibatch size, then start from the seed again.
    EquationSampler::new(training_equations, options.seed)?.sample(batch)?;
    let mut sampler = EquationSampler::new(training_equations, options.seed)?;
    if let Some(epochs) = options.epochs {
        options.steps = epoch_steps(epochs, training_equations, batch)?;
    }
    let steps_per_epoch = training_equations.div_ceil(batch);
    let init_seed = options.init_seed.unwrap_or(options.seed);
    let mut model = GpuSimpleModel::new(options.device, config, init_seed)?;
    let mut splits = Vec::with_capacity(2);
    for validation in [false, true] {
        let (features, targets) = config.encode(dataset.split(validation))?;
        splits.push(Split {
            table: model.upload_table(&features, &targets)?,
            targets,
        });
    }
    let parameters = config.parameter_count();
    std::fs::create_dir_all(&options.output)?;
    write_dataset(&options.output, &dataset)?;
    let applicable = |used: bool, value: usize| {
        if used {
            value.to_string()
        } else {
            "none".into()
        }
    };
    std::fs::write(
        options.output.join("run.txt"),
        format!(
            "paper=https://arxiv.org/abs/2201.02177v1\nstatus=running\nmodel=simple\nfamily={}\narchitecture={}\nactivation={}\nencoding={}\noperation={}\nmodulus={}\ntrain_fraction={}\ntraining_equations={}\nvalidation_equations={}\nseed={}\ninit_seed={init_seed}\ninputs=residue_values\nabstract_symbol_permutation=false\nloss=answer_only\noutput_classes={}\ninput_features={}\nwidth={}\ndepth={}\nexpansion={}\nconnectivity={}\nmesh_layers={}\ntied_mesh={}\nparameters={parameters}\nall_parameters={parameters}\nnormalization={}\ninitialization={}\nbias=false\nbetas=0.9,0.98\nepsilon=1e-8\nlearning_rate={}\nweight_decay={}\ndecay_factors={}\nwarmup_steps=10\nbatch={batch}\nsteps={}\nreport_every={}\nevaluation=complete_splits\ncheckpoint={}\nrequested_epochs={}\nsteps_per_epoch={steps_per_epoch}\nepoch_definition=one_visit_to_every_training_equation\nepoch_tail=short_batch\nstop_validation_accuracy={}\n",
            family(config),
            config.architecture,
            match config.architecture {
                Architecture::Bilinear => "none".into(),
                _ => config.activation.to_string(),
            },
            config.encoding,
            options.operation,
            config.modulus,
            options.train_fraction,
            training_equations,
            dataset.split(true).len(),
            options.seed,
            config.modulus,
            config.input_features(),
            config.width,
            applicable(config.architecture != Architecture::Bilinear, config.depth),
            applicable(
                config.architecture == Architecture::ResidualMlp,
                config.expansion
            ),
            match config.architecture {
                Architecture::MeshMlp => config.connectivity.name(),
                _ => "none",
            },
            config.mesh_layers(),
            match config.architecture {
                Architecture::MeshMlp => config.tied_mesh.to_string(),
                _ => "none".into(),
            },
            match config.architecture {
                Architecture::ResidualMlp => "pre_rmsnorm_blocks_and_final",
                _ => "none",
            },
            match config.architecture {
                Architecture::MeshMlp => "uniform_fan_in_dense_identity_mesh",
                _ => "uniform_fan_in",
            },
            options.optimizer.learning_rate,
            options.optimizer.weight_decay,
            config.decay_factors,
            options.steps,
            options.report_every,
            match config.architecture {
                Architecture::MeshMlp if config.tied_mesh => {
                    "raw_f32_le_dense_layers_then_one_tied_mesh_layer"
                }
                Architecture::MeshMlp => "raw_f32_le_dense_layers_then_mesh_layers",
                _ => "raw_f32_le_layer_order",
            },
            options
                .epochs
                .map_or_else(|| "none".into(), |n| n.to_string()),
            options
                .stop_validation_accuracy
                .map_or_else(|| "none".into(), |x| x.to_string()),
        ),
    )?;
    println!(
        "Simple-model grokking: {} {} mod {}, train={}, validation={}, batch={}, parameters={parameters}",
        family(config),
        options.operation,
        config.modulus,
        training_equations,
        dataset.split(true).len(),
        batch,
    );
    let stop = options
        .stop_validation_accuracy
        .map_or_else(String::new, |x| format!(" or validation accuracy >={x}"));
    let factors = match config.decay_factors {
        2 => String::new(),
        n => format!(" as {n} factors per channel"),
    };
    println!(
        "Answer-only loss; AdamW beta2=0.98, decay={}{factors}, warmup=10; continuing for {} updates{stop}",
        options.optimizer.weight_decay, options.steps
    );
    let mut history = BufWriter::new(File::create(options.output.join("loss.csv"))?);
    writeln!(
        history,
        "step,training_examples_seen,completed_epochs,examples_in_partial_epoch,train_cross_entropy,validation_cross_entropy,train_accuracy,validation_accuracy,weight_l2,learning_rate,elapsed_seconds"
    )?;
    let start = Instant::now();
    let mut thresholds = Thresholds::default();
    let mut validation_accuracy = record(
        &mut history,
        &mut model,
        &splits,
        &options.output,
        [0, 0],
        0.0,
        start.elapsed().as_secs_f64(),
        &mut thresholds,
    )?;
    std::fs::copy(
        options.output.join("model.bin"),
        options.output.join("initial-model.bin"),
    )?;
    let (mut step, mut trained_examples) = (0, 0);
    let mut epoch = Epoch::sample(
        &model,
        &splits[0],
        &mut sampler,
        batch,
        steps_per_epoch.min(options.steps),
    )?;
    // The stopping rule only sees complete-split evaluations, never training batches.
    while step < options.steps
        && !reached_stop(options.stop_validation_accuracy, validation_accuracy)
    {
        if epoch.next == epoch.minibatches.len() {
            epoch = Epoch::sample(
                &model,
                &splits[0],
                &mut sampler,
                batch,
                steps_per_epoch.min(options.steps - step),
            )?;
        }
        let [offset, rows] = epoch.minibatches[epoch.next];
        epoch.next += 1;
        step += 1;
        let optimizer = warmed_optimizer(options.optimizer, step);
        model.train_batch(&splits[0].table, &epoch.batches, offset, rows, optimizer)?;
        trained_examples += rows;
        if step.is_multiple_of(options.report_every) || step == options.steps {
            validation_accuracy = record(
                &mut history,
                &mut model,
                &splits,
                &options.output,
                [step, trained_examples],
                optimizer.learning_rate,
                start.elapsed().as_secs_f64(),
                &mut thresholds,
            )?;
        }
    }
    let stop_reason = if reached_stop(options.stop_validation_accuracy, validation_accuracy) {
        "validation_accuracy"
    } else {
        "update_budget"
    };
    write_predictions(&options.output, &mut model, &dataset, &splits)?;
    let metadata_path = options.output.join("run.txt");
    let metadata =
        std::fs::read_to_string(&metadata_path)?.replace("status=running", "status=complete");
    std::fs::write(&metadata_path, metadata)?;
    let mut metadata = OpenOptions::new().append(true).open(&metadata_path)?;
    writeln!(
        metadata,
        "elapsed_seconds={:.6}\nstop_reason={stop_reason}\ncompleted_steps={step}\ntraining_examples_seen={}\ncompleted_epochs={}\nexamples_in_partial_epoch={}\nfirst_recorded_train_99_step={}\nfirst_recorded_validation_99_step={}",
        start.elapsed().as_secs_f64(),
        trained_examples,
        trained_examples / training_equations,
        trained_examples % training_equations,
        observed(thresholds.training),
        observed(thresholds.validation)
    )?;
    println!(
        "Saved weights, split, predictions, and learning curves to {}",
        options.output.display()
    );
    println!(
        "First recorded accuracy >=99%: train={}, validation={}",
        observed(thresholds.training),
        observed(thresholds.validation)
    );
    println!("Stopped after {step} updates: {stop_reason}");
    Ok(())
}

/// Groups runs whose size alone varies, such as MLP widths and depths.
fn family(config: SimpleConfig) -> String {
    match config.architecture {
        Architecture::Bilinear => format!("{}-{}", config.architecture, config.encoding),
        Architecture::MeshMlp => format!(
            "{}-{}{}-{}-{}",
            config.architecture,
            config.connectivity,
            if config.tied_mesh { "-tied" } else { "" },
            config.activation,
            config.encoding
        ),
        _ if config.decay_factors != 2 => format!(
            "{}-{}-{}-decay-factors-{}",
            config.architecture, config.activation, config.encoding, config.decay_factors
        ),
        _ => format!(
            "{}-{}-{}",
            config.architecture, config.activation, config.encoding
        ),
    }
}

struct Metrics {
    loss: f64,
    accuracy: f64,
}

/// Cross-entropy and prediction of every row, as for Mesh GPT answers.
fn row_results(model: &mut GpuSimpleModel, split: &Split) -> Result<Vec<(f64, usize)>> {
    let logits = model.logits(&split.table)?;
    Ok(logits
        .chunks_exact(model.config().modulus)
        .zip(&split.targets)
        .map(|(row, &target)| row_metrics(row, target as usize))
        .collect())
}

fn evaluate(model: &mut GpuSimpleModel, split: &Split) -> Result<Metrics> {
    let rows = row_results(model, split)?;
    let (mut loss, mut correct) = (0.0, 0);
    for ((cross_entropy, prediction), &target) in rows.iter().zip(&split.targets) {
        loss += cross_entropy;
        correct += usize::from(*prediction == target as usize);
    }
    Ok(Metrics {
        loss: loss / rows.len() as f64,
        accuracy: correct as f64 / rows.len() as f64,
    })
}

/// `progress` holds the completed updates and the training examples they saw.
#[allow(clippy::too_many_arguments)]
fn record(
    history: &mut impl Write,
    model: &mut GpuSimpleModel,
    splits: &[Split],
    directory: &Path,
    progress: [usize; 2],
    learning_rate: f32,
    elapsed: f64,
    thresholds: &mut Thresholds,
) -> Result<f64> {
    let [step, trained_examples] = progress;
    let train = evaluate(model, &splits[0])?;
    let validation = evaluate(model, &splits[1])?;
    if train.accuracy >= 0.99 && thresholds.training.is_none() {
        thresholds.training = Some(step);
    }
    if validation.accuracy >= 0.99 && thresholds.validation.is_none() {
        thresholds.validation = Some(step);
    }
    let parameters = model.parameters()?;
    let norm = parameters
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    let count = splits[0].targets.len();
    writeln!(
        history,
        "{step},{trained_examples},{},{},{:.9},{:.9},{:.9},{:.9},{norm:.9},{learning_rate:.9},{elapsed:.6}",
        trained_examples / count,
        trained_examples % count,
        train.loss,
        validation.loss,
        train.accuracy,
        validation.accuracy
    )?;
    history.flush()?;
    let bytes: Vec<_> = parameters.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(directory.join("model.bin"), bytes)?;
    println!(
        "step {step:>7}: train CE={:.5} accuracy={:.2}%; validation CE={:.5} accuracy={:.2}%",
        train.loss,
        100.0 * train.accuracy,
        validation.loss,
        100.0 * validation.accuracy
    );
    Ok(validation.accuracy)
}

fn write_predictions(
    directory: &Path,
    model: &mut GpuSimpleModel,
    dataset: &EquationDataset,
    splits: &[Split],
) -> Result<()> {
    let mut file = BufWriter::new(File::create(directory.join("predictions.csv"))?);
    writeln!(file, "split,a,b,answer,prediction,correct,cross_entropy")?;
    for (validation, split) in [false, true].into_iter().zip(splits) {
        let rows = row_results(model, split)?;
        for (equation, (loss, prediction)) in dataset.split(validation).iter().zip(rows) {
            writeln!(
                file,
                "{},{},{},{},{prediction},{},{loss:.9}",
                if validation { "validation" } else { "train" },
                equation.a,
                equation.b,
                equation.answer,
                usize::from(prediction == equation.answer)
            )?;
        }
    }
    file.flush()?;
    Ok(())
}
