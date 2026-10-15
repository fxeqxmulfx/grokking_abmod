use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, Sender, SyncSender},
    time::Instant,
};

use cuda_core::PinnedHostBuffer;
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
    verify_model: Option<Architecture>,
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
            verify_model: None,
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
                "--mesh-depth" => options.config.mesh_depth = value(&mut arguments, &flag)?,
                "--mesh-output-activation" => options.config.mesh_output_activation = true,
                "--connectivity" => options.config.connectivity = value(&mut arguments, &flag)?,
                "--tied-mesh" => options.config.tied_mesh = true,
                "--mesh-initialization" => {
                    options.config.full_mesh_initialization = value(&mut arguments, &flag)?
                }
                "--embedding-mesh-depth" => {
                    options.config.embedding_mesh_depth = value(&mut arguments, &flag)?
                }
                "--embedding-initialization" => {
                    options.config.shared_embedding_initialization = value(&mut arguments, &flag)?
                }
                "--dense-embedding-initialization" => {
                    options.config.dense_embedding_initialization = value(&mut arguments, &flag)?
                }
                "--decay-factors" => options.config.decay_factors = value(&mut arguments, &flag)?,
                "--mesh-learning-rate-multiplier" => {
                    options.config.mesh_learning_rate_multiplier = value(&mut arguments, &flag)?
                }
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
                "--verify-model" => {
                    options.verify_model = Some(value(&mut arguments, &flag)?);
                    options.verify = true;
                }
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
            --model NAME          mlp (default), mesh-mlp, full-mesh-mlp, shared-neighbor-mesh-mlp,\n\
                                  tied-concat-mesh-mlp,\n\
                                  residual-mlp, residual-mesh-mlp,\n\
                                  tied-concat-residual-mesh-mlp,\n\
                                  tied-rotary-residual-mesh-mlp, tied-rotary-residual-dense-mlp, or bilinear\n\
            --encoding NAME       Per operand: one-hot (default), binary (+-1 bits),\n\
                                  scalar ((v-48)/48 for p=97), or discrete-log\n\
            --mesh-initialization Full mesh only: kaiming (default), identity-hidden\n\
            --embedding-mesh-depth Shared Neighbor embedding layers (default 4)\n\
            --embedding-initialization Shared Neighbor: near-identity (default),\n\
                                  unit-norm-weak, or unit-norm-strong\n\
            --dense-embedding-initialization Rotary Dense: uniform-fan-in (default),\n\
                                  or normal-width (variance 1/width)\n\
            --verify-model NAME   Verify only this architecture and its launch contracts\n\
                                  ([v==0, centered log to a primitive root]; prime p)\n\
            --activation NAME     relu (default) or square; bilinear has none\n\
            --width N             MLP hidden width, residual stream, or bilinear rank (default: 128)\n\
            --depth N             MLP hidden layers or residual blocks, 1..64 (default: 1);\n\
                                  a mesh MLP's layers after the first are mesh layers\n\
            --expansion N         Residual block hidden width / stream width (default: 4)\n\
            --mesh-depth N        Residual mesh branch layers, 1..64 (default: 4)\n\
            --mesh-output-activation  Apply the activation after the final mesh, before the skip\n\
            --connectivity NAME   Mesh MLP second parent: neighbor, c+1 (default),\n\
                                  butterfly, c+2^(layer mod floor(log2 width)), or\n\
                                  own, none: each mesh unit scales its own channel\n\
            --tied-mesh           Mesh MLP: every mesh layer uses one stored layer's weights\n\
            --decay-factors N     One-layer MLP: decay channel c's input column and readout\n\
                                  row at decay*s_c^(2/N-1), s_c=(|E_c|^2+|R_c|^2)/2, the\n\
                                  balanced L2 penalty of N factors (default: 2, plain AdamW)\n\
            --mesh-learning-rate-multiplier K  Mesh MLP: mesh-layer weights train at K times\n\
                                  the learning rate, decay included (default: 1)\n\
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
        return match options.verify_model {
            Some(model) => GpuSimpleModel::verify_architecture(options.device, model),
            None => GpuSimpleModel::verify(options.device),
        };
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
            "paper=https://arxiv.org/abs/2201.02177v1\nstatus=running\nmodel=simple\nfamily={}\narchitecture={}\nactivation={}\nencoding={}\noperation={}\nmodulus={}\ntrain_fraction={}\ntraining_equations={}\nvalidation_equations={}\nseed={}\ninit_seed={init_seed}\ninputs=residue_values\nabstract_symbol_permutation=false\nloss=answer_only\noutput_classes={}\ninput_features={}\nwidth={}\ndepth={}\nexpansion={}\nconnectivity={}\nmesh_layers={}\ntied_mesh={}\nparameters={parameters}\nall_parameters={parameters}\nnormalization={}\ninitialization={}\nbias={}\nbetas=0.9,0.98\nepsilon=1e-8\nlearning_rate={}\nmesh_learning_rate_multiplier={}\nweight_decay={}\ndecay_factors={}\nwarmup_steps=10\nbatch={batch}\nsteps={}\nreport_every={}\nevaluation=complete_splits\ncheckpoint={}\nrequested_epochs={}\nsteps_per_epoch={steps_per_epoch}\nepoch_definition=one_visit_to_every_training_equation\nepoch_tail=short_batch\nstop_validation_accuracy={}\n",
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
                Architecture::MeshMlp
                | Architecture::FullMeshMlp
                | Architecture::SharedNeighborMeshMlp
                | Architecture::TiedConcatMeshMlp
                | Architecture::TiedConcatResidualMeshMlp
                | Architecture::TiedRotaryResidualMeshMlp
                | Architecture::ResidualMeshMlp => config.connectivity.name(),
                _ => "none",
            },
            config.mesh_layers(),
            match config.architecture {
                Architecture::MeshMlp
                | Architecture::FullMeshMlp
                | Architecture::SharedNeighborMeshMlp
                | Architecture::TiedConcatMeshMlp
                | Architecture::TiedConcatResidualMeshMlp
                | Architecture::TiedRotaryResidualMeshMlp
                | Architecture::ResidualMeshMlp => config.tied_mesh.to_string(),
                _ => "none".into(),
            },
            match config.architecture {
                Architecture::ResidualMlp
                | Architecture::ResidualMeshMlp
                | Architecture::TiedConcatResidualMeshMlp
                | Architecture::TiedRotaryResidualMeshMlp
                | Architecture::TiedRotaryResidualDenseMlp
                | Architecture::SharedNeighborMeshMlp => "pre_rmsnorm_blocks_and_final",
                _ => "none",
            },
            match config.architecture {
                Architecture::MeshMlp | Architecture::ResidualMeshMlp =>
                    "uniform_fan_in_dense_identity_mesh",
                Architecture::TiedConcatMeshMlp | Architecture::TiedConcatResidualMeshMlp =>
                    "uniform_fan_in_embedding_projection_identity_mesh",
                Architecture::TiedRotaryResidualMeshMlp => "uniform_fan_in_embedding_identity_mesh",
                Architecture::TiedRotaryResidualDenseMlp =>
                    match config.dense_embedding_initialization {
                        grokking_abmod::simple::DenseEmbeddingInitialization::UniformFanIn =>
                            "uniform_fan_in_embedding_kaiming_normal_relu_gain_dense",
                        grokking_abmod::simple::DenseEmbeddingInitialization::NormalWidth =>
                            "normal_width_embedding_kaiming_normal_relu_gain_dense",
                    },
                Architecture::FullMeshMlp => match config.full_mesh_initialization {
                    grokking_abmod::simple::FullMeshInitialization::Kaiming =>
                        "kaiming_normal_two_parent_zero_bias",
                    grokking_abmod::simple::FullMeshInitialization::IdentityHidden =>
                        "kaiming_normal_input_output_identity_hidden_zero_bias",
                },
                Architecture::SharedNeighborMeshMlp => {
                    if config.shared_embedding_initialization.normalizes_columns() {
                        "unit_column_norm_rotation_factors_shared_neighbor_kaiming_or_identity_core"
                    } else {
                        "near_identity_gaussian_shared_neighbor_kaiming_or_identity_core"
                    }
                }
                _ => "uniform_fan_in",
            },
            matches!(
                config.architecture,
                Architecture::FullMeshMlp | Architecture::SharedNeighborMeshMlp
            ),
            options.optimizer.learning_rate,
            match config.architecture {
                Architecture::MeshMlp
                | Architecture::FullMeshMlp
                | Architecture::SharedNeighborMeshMlp
                | Architecture::TiedConcatMeshMlp
                | Architecture::TiedConcatResidualMeshMlp
                | Architecture::TiedRotaryResidualMeshMlp
                | Architecture::ResidualMeshMlp => config.mesh_learning_rate_multiplier.to_string(),
                _ => "none".into(),
            },
            options.optimizer.weight_decay,
            config.decay_factors,
            options.steps,
            options.report_every,
            match config.architecture {
                Architecture::MeshMlp if config.tied_mesh => {
                    "raw_f32_le_dense_layers_then_one_tied_mesh_layer"
                }
                Architecture::MeshMlp | Architecture::ResidualMeshMlp =>
                    "raw_f32_le_dense_layers_then_mesh_layers",
                Architecture::TiedConcatMeshMlp | Architecture::TiedConcatResidualMeshMlp =>
                    "raw_f32_le_shared_embedding_projection_then_stored_mesh_layers",
                Architecture::TiedRotaryResidualMeshMlp =>
                    "raw_f32_le_shared_embedding_then_mesh_layers",
                Architecture::TiedRotaryResidualDenseMlp =>
                    "raw_f32_le_shared_embedding_then_dense_branches",
                Architecture::FullMeshMlp => "raw_f32_le_two_parent_layers_unit_weights_then_bias",
                Architecture::SharedNeighborMeshMlp =>
                    "raw_f32_le_shared_neighbor_layers_then_two_parent_core",
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
    if config.architecture == Architecture::TiedRotaryResidualDenseMlp {
        let mut metadata = OpenOptions::new()
            .append(true)
            .open(options.output.join("run.txt"))?;
        let variance = match config.dense_embedding_initialization {
            grokking_abmod::simple::DenseEmbeddingInitialization::UniformFanIn => {
                1.0 / (3 * config.modulus) as f64
            }
            grokking_abmod::simple::DenseEmbeddingInitialization::NormalWidth => {
                1.0 / config.width as f64
            }
        };
        writeln!(
            metadata,
            "dense_embedding_initialization={}\nembedding_initial_variance={variance:.12}\nembedding_scale_constraint=initialization_only\ndense_branch_initialization=kaiming_normal_variance_2_over_width",
            config.dense_embedding_initialization
        )?;
    }
    if config.architecture == Architecture::FullMeshMlp {
        let mut metadata = OpenOptions::new()
            .append(true)
            .open(options.output.join("run.txt"))?;
        writeln!(
            metadata,
            "max_activation_parents=2\ninput_projection=two_parent\nreadout=two_parent\nmesh_initialization={}",
            config.full_mesh_initialization
        )?;
        let receptive_fields = config.full_mesh_receptive_fields();
        writeln!(
            metadata,
            "readout_min_input_features={}\nreadout_max_input_features={}",
            receptive_fields
                .iter()
                .map(|field| field.count_ones())
                .min()
                .unwrap_or(0),
            receptive_fields
                .iter()
                .map(|field| field.count_ones())
                .max()
                .unwrap_or(0)
        )?;
    }
    if config.architecture == Architecture::SharedNeighborMeshMlp {
        let mut metadata = OpenOptions::new()
            .append(true)
            .open(options.output.join("run.txt"))?;
        writeln!(
            metadata,
            "max_activation_parents=2\nembedding_mesh_depth={}\nembedding_connectivity=neighbor\nembedding_weight_sharing=a,b,transposed_readout\noperand_rotation=pairwise_b_plus_90_degrees\nreadout=transposed_neighbor_chain\nmesh_initialization={}",
            config.embedding_mesh_depth, config.full_mesh_initialization
        )?;
        let mode = config.shared_embedding_initialization;
        writeln!(
            metadata,
            "shared_embedding_initialization={mode}\nembedding_initialization={}\nembedding_mixing_strength={:.12}\nembedding_column_norm_target={}\nembedding_column_normalization={}",
            if mode.normalizes_columns() {
                "paired_rotation_factors_with_unit_implicit_columns"
            } else {
                "identity_plus_gaussian_mixing_std_0.1_over_sqrt_depth"
            },
            mode.ratio(config.embedding_mesh_depth),
            if mode.normalizes_columns() {
                "1"
            } else {
                "none"
            },
            if mode.normalizes_columns() {
                "initialization_only"
            } else {
                "none"
            },
        )?;
        writeln!(
            metadata,
            "core_structure=pre_rmsnorm_mesh_activation_residual\nfinal_rmsnorm=true\nrmsnorm_epsilon=1e-5\ninput_coverage_excludes_norm_statistics=true"
        )?;
        let coverage = config.shared_neighbor_input_coverage();
        writeln!(
            metadata,
            "readout_min_input_features={}\nreadout_max_input_features={}",
            coverage.iter().min().unwrap_or(&0),
            coverage.iter().max().unwrap_or(&0)
        )?;
    }
    println!(
        "Simple-model grokking: {} {} mod {}, train={}, validation={}, batch={}, parameters={parameters}",
        family(config),
        options.operation,
        config.modulus,
        training_equations,
        dataset.split(true).len(),
        batch,
    );
    if config.architecture.residual_mesh() {
        let mut metadata = OpenOptions::new()
            .append(true)
            .open(options.output.join("run.txt"))?;
        writeln!(metadata, "mesh_depth={}", config.mesh_depth)?;
        writeln!(
            metadata,
            "mesh_output_activation={}",
            config.mesh_output_activation
        )?;
        writeln!(
            metadata,
            "pointwise_activations_per_branch={}",
            config.mesh_depth - 1 + usize::from(config.mesh_output_activation)
        )?;
        if config.architecture.rotary_embeddings() {
            writeln!(metadata, "operand_rotation=pairwise_b_plus_90_degrees")?;
            writeln!(metadata, "embedding_roles=a,b,readout")?;
        }
    }
    let stop = options
        .stop_validation_accuracy
        .map_or_else(String::new, |x| format!(" or validation accuracy >={x}"));
    let factors = match config.decay_factors {
        2 => String::new(),
        n => format!(" as {n} factors per channel"),
    };
    let mesh_rate = match config.mesh_learning_rate_multiplier {
        1 => String::new(),
        k => format!(", mesh layers at {k}x the rate"),
    };
    println!(
        "Answer-only loss; AdamW beta2=0.98, decay={}{factors}, warmup=10{mesh_rate}; continuing for {} updates{stop}",
        options.optimizer.weight_decay, options.steps
    );
    let mut history = BufWriter::new(File::create(options.output.join("loss.csv"))?);
    writeln!(
        history,
        "step,training_examples_seen,completed_epochs,examples_in_partial_epoch,train_cross_entropy,validation_cross_entropy,train_accuracy,validation_accuracy,weight_l2,learning_rate,elapsed_seconds"
    )?;
    let start = Instant::now();
    let mut thresholds = Thresholds::default();
    let mut validation_accuracy = 0.0;
    let (mut step, mut trained_examples) = (0, 0);
    let targets = [splits[0].targets.as_slice(), splits[1].targets.as_slice()];
    let (directory, classes) = (options.output.as_path(), model.config().modulus);
    // A writer thread scores and saves each record while training continues,
    // then returns its page-locked buffers for later records.
    std::thread::scope(|scope| -> Result<()> {
        let (records, queue) = mpsc::sync_channel(2);
        let (returns, returned) = mpsc::channel();
        let writer = scope.spawn(move || {
            write_records(queue, returns, history, targets, directory, classes)
                .map_err(|error| error.to_string())
        });
        let records = (records, returned);
        let trained = (|| -> Result<()> {
            validation_accuracy = record(
                &records,
                &mut model,
                &splits,
                [0, 0],
                0.0,
                start.elapsed().as_secs_f64(),
                &mut thresholds,
            )?;
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
                        &records,
                        &mut model,
                        &splits,
                        [step, trained_examples],
                        optimizer.learning_rate,
                        start.elapsed().as_secs_f64(),
                        &mut thresholds,
                    )?;
                }
            }
            Ok(())
        })();
        drop(records);
        // A writer error explains a failed send, so it comes first.
        writer.join().map_err(|_| "the record writer panicked")??;
        trained
    })?;
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
        Architecture::MeshMlp
        | Architecture::FullMeshMlp
        | Architecture::SharedNeighborMeshMlp
        | Architecture::TiedConcatMeshMlp
        | Architecture::ResidualMeshMlp
        | Architecture::TiedConcatResidualMeshMlp
        | Architecture::TiedRotaryResidualMeshMlp => {
            format!(
                "{}-{}{}-{}-{}{}{}",
                config.architecture,
                config.connectivity,
                if config.tied_mesh { "-tied" } else { "" },
                config.activation,
                config.encoding,
                match config.mesh_learning_rate_multiplier {
                    1 => String::new(),
                    k => format!("-mesh-rate-{k}"),
                },
                if config.mesh_output_activation {
                    "-activated-output"
                } else {
                    ""
                },
            )
        }
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

/// Cross-entropy and prediction of every row, as for Mesh GPT answers.
/// Rows are independent: worker threads score contiguous chunks, and the
/// results keep the row order, so every later sum is unchanged.
fn score_rows(logits: &[f32], targets: &[u32], classes: usize) -> Result<Vec<(f64, usize)>> {
    let rows = targets.len();
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    let chunk = rows.div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        let workers: Vec<_> = logits
            .chunks(chunk * classes)
            .zip(targets.chunks(chunk))
            .map(|(logits, targets)| {
                scope.spawn(move || {
                    logits
                        .chunks_exact(classes)
                        .zip(targets)
                        .map(|(row, &target)| row_metrics(row, target as usize))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut results = Vec::with_capacity(rows);
        for worker in workers {
            results.extend(
                worker
                    .join()
                    .map_err(|_| "row evaluation thread panicked")?,
            );
        }
        Ok(results)
    })
}

fn row_results(model: &mut GpuSimpleModel, split: &Split) -> Result<Vec<(f64, usize)>> {
    let logits = model.logits(&split.table)?;
    score_rows(&logits, &split.targets, model.config().modulus)
}

/// The fraction of rows whose prediction is the target.
fn split_accuracy(predictions: &[u32], targets: &[u32]) -> f64 {
    let correct = predictions
        .iter()
        .zip(targets)
        .filter(|(prediction, target)| prediction == target)
        .count();
    correct as f64 / targets.len() as f64
}

/// The device results of one record, which the writer thread scores.
struct Snapshot {
    /// The completed updates and the training examples they saw.
    progress: [usize; 2],
    learning_rate: f32,
    elapsed: f64,
    /// Logits and accuracy of the training and validation splits.
    logits: [PinnedHostBuffer<f32>; 2],
    accuracy: [f64; 2],
    parameters: PinnedHostBuffer<f32>,
}

/// Evaluates both splits on the device and queues the record. Returns the
/// validation accuracy, the only record value that can stop training.
/// `records` also carries back the buffers of written records.
fn record(
    records: &(SyncSender<Snapshot>, Receiver<PinnedHostBuffer<f32>>),
    model: &mut GpuSimpleModel,
    splits: &[Split],
    progress: [usize; 2],
    learning_rate: f32,
    elapsed: f64,
    thresholds: &mut Thresholds,
) -> Result<f64> {
    let (records, returned) = records;
    for buffer in returned.try_iter() {
        model.reuse(buffer);
    }
    let step = progress[0];
    let evaluations = model.evaluate([&splits[0].table, &splits[1].table])?;
    let accuracy = [0, 1].map(|i| split_accuracy(&evaluations[i].predictions, &splits[i].targets));
    let logits = evaluations.map(|evaluation| evaluation.logits);
    if accuracy[0] >= 0.99 && thresholds.training.is_none() {
        thresholds.training = Some(step);
    }
    if accuracy[1] >= 0.99 && thresholds.validation.is_none() {
        thresholds.validation = Some(step);
    }
    let snapshot = Snapshot {
        progress,
        learning_rate,
        elapsed,
        logits,
        accuracy,
        parameters: model.pinned_parameters()?,
    };
    records
        .send(snapshot)
        .map_err(|_| "the record writer stopped")?;
    Ok(accuracy[1])
}

/// The mean cross-entropy, summed in row order.
fn mean_cross_entropy(logits: &[f32], targets: &[u32], classes: usize) -> Result<f64> {
    let rows = score_rows(logits, targets, classes)?;
    let mut loss = 0.0;
    for (cross_entropy, _) in &rows {
        loss += cross_entropy;
    }
    Ok(loss / rows.len() as f64)
}

/// Writes records in order: a `loss.csv` line, `model.bin` and a console
/// line each. The first record's parameters also become `initial-model.bin`.
/// Each record's buffers then go back through `returns`.
fn write_records(
    records: Receiver<Snapshot>,
    returns: Sender<PinnedHostBuffer<f32>>,
    mut history: impl Write,
    targets: [&[u32]; 2],
    directory: &Path,
    classes: usize,
) -> Result<()> {
    for snapshot in records {
        let Snapshot {
            progress: [step, trained_examples],
            learning_rate,
            elapsed,
            logits,
            accuracy,
            parameters,
        } = snapshot;
        let train_loss = mean_cross_entropy(&logits[0], targets[0], classes)?;
        let validation_loss = mean_cross_entropy(&logits[1], targets[1], classes)?;
        let norm = parameters
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        let count = targets[0].len();
        writeln!(
            history,
            "{step},{trained_examples},{},{},{:.9},{:.9},{:.9},{:.9},{norm:.9},{learning_rate:.9},{elapsed:.6}",
            trained_examples / count,
            trained_examples % count,
            train_loss,
            validation_loss,
            accuracy[0],
            accuracy[1]
        )?;
        history.flush()?;
        let bytes: Vec<_> = parameters.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(directory.join("model.bin"), bytes)?;
        if step == 0 {
            std::fs::copy(
                directory.join("model.bin"),
                directory.join("initial-model.bin"),
            )?;
        }
        println!(
            "step {step:>7}: train CE={train_loss:.5} accuracy={:.2}%; validation CE={validation_loss:.5} accuracy={:.2}%",
            100.0 * accuracy[0],
            100.0 * accuracy[1]
        );
        // Once training has ended, a failed send just drops the buffer.
        for buffer in logits.into_iter().chain([parameters]) {
            let _ = returns.send(buffer);
        }
    }
    Ok(())
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
