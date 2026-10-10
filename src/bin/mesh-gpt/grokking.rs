use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use grokking_abmod::{
    Result,
    gpt::{
        AdamWOptions, Checkpoint, EQUATION_CONTEXT, EquationDataset, EquationSampler, GptConfig,
        ModularOperation, SpecializedMeshGpt,
    },
};

use super::value;

type GrokModel = SpecializedMeshGpt<128, 256, EQUATION_CONTEXT, 8>;

struct Options {
    config: GptConfig,
    optimizer: AdamWOptions,
    modulus: usize,
    operation: ModularOperation,
    train_fraction: f64,
    steps: usize,
    batch: Option<usize>,
    report_every: usize,
    seed: u64,
    device: usize,
    output: PathBuf,
    help: bool,
}

impl Options {
    fn parse(mut arguments: impl Iterator<Item = String>) -> Result<Self> {
        let mut options = Self {
            config: GptConfig {
                width: 128,
                context: EQUATION_CONTEXT,
                ..GptConfig::default()
            },
            optimizer: AdamWOptions {
                learning_rate: 0.001,
                weight_decay: 1.0,
                beta2: 0.98,
                ..AdamWOptions::default()
            },
            modulus: 97,
            operation: ModularOperation::Add,
            train_fraction: 0.5,
            steps: 100_000,
            batch: None,
            report_every: 1000,
            seed: 7,
            device: 0,
            output: PathBuf::from("target/mesh-gpt-grokking"),
            help: false,
        };
        while let Some(flag) = arguments.next() {
            match flag.as_str() {
                "--modulus" => options.modulus = value(&mut arguments, &flag)?,
                "--operation" => options.operation = value(&mut arguments, &flag)?,
                "--train-fraction" => options.train_fraction = value(&mut arguments, &flag)?,
                "--steps" => options.steps = value(&mut arguments, &flag)?,
                "--batch" => options.batch = Some(value(&mut arguments, &flag)?),
                "--report-every" => options.report_every = value(&mut arguments, &flag)?,
                "--width" => options.config.width = value(&mut arguments, &flag)?,
                "--heads" => options.config.heads = value(&mut arguments, &flag)?,
                "--layers" => options.config.layers = value(&mut arguments, &flag)?,
                "--attention-depth" => {
                    options.config.attention_mesh_depth = value(&mut arguments, &flag)?
                }
                "--ffn-depth" => options.config.ffn_mesh_depth = value(&mut arguments, &flag)?,
                "--learning-rate" => {
                    options.optimizer.learning_rate = value(&mut arguments, &flag)?
                }
                "--weight-decay" => options.optimizer.weight_decay = value(&mut arguments, &flag)?,
                "--seed" => options.seed = value(&mut arguments, &flag)?,
                "--device" => options.device = value(&mut arguments, &flag)?,
                "--output" => options.output = value(&mut arguments, &flag)?,
                "--help" | "-h" => options.help = true,
                _ => return Err(format!("unknown grok option: {flag}; use grok --help").into()),
            }
        }
        Ok(options)
    }
}

pub fn run(arguments: impl Iterator<Item = String>) -> Result<()> {
    let mut options = Options::parse(arguments)?;
    if options.help {
        println!(
            "Mesh GPT grokking: abstract-symbol modular equations from arXiv:2201.02177v1\n\
            Usage: cargo oxide run --bin mesh-gpt -- grok [options]\n\n\
            --operation NAME      add (default), sub, div, mixed, square-sum, quadratic,\n\
                                  quadratic-linear, cubic-product, cubic-mixed\n\
            --modulus P           Prime, 2..254 (default: 97)\n\
            --train-fraction X    Fixed training fraction, 0<X<1 (default: 0.5)\n\
            --steps N             Gradient updates (default: 100000; no early stopping)\n\
            --batch N             Equations per minibatch (default: min(512, train_size/2))\n\
            --report-every N      Evaluate both complete splits (default: 1000)\n\
            --learning-rate X     Maximum AdamW rate (default: 0.001; 10-step warmup)\n\
            --weight-decay X      AdamW decay (default: 1; betas fixed at 0.9, 0.98)\n\
            --width N             Channels, 2..128 (default: 128)\n\
            --heads N             Heads, 1..8, even channels/head (default: 4)\n\
            --layers N            Transformer blocks, 1..64 (default: 2)\n\
            --attention-depth N   Embedding/Q/K/V/output mesh depth (default: 4)\n\
            --ffn-depth N         Feed-forward mesh depth (default: 6)\n\
            --seed N              Model, symbol permutation, split, sampling seed (default: 7)\n\
            --device N            CUDA device ordinal (default: 0)\n\
            --output PATH         Artifacts (default: target/mesh-gpt-grokking)"
        );
        return Ok(());
    }
    if options.steps == 0 || options.report_every == 0 {
        return Err("steps and report interval must be positive".into());
    }
    options.optimizer.validate()?;
    let dataset = EquationDataset::new(
        options.modulus,
        options.operation,
        options.train_fraction,
        options.seed,
    )?;
    options.config.vocab_size = dataset.tokenizer().vocab_size();
    options.config.validate()?;
    let batch = options
        .batch
        .unwrap_or_else(|| 512.min(dataset.split(false).len() / 2));
    let mut sampler = EquationSampler::new(dataset.split(false).len(), options.seed)?;
    sampler.sample(batch)?;
    // Start from the seed again after validating the requested minibatch size.
    let mut sampler = EquationSampler::new(dataset.split(false).len(), options.seed)?;
    let mut model = GrokModel::new(options.device, options.config, options.seed)?;
    std::fs::create_dir_all(&options.output)?;
    write_dataset(&options.output, &dataset)?;
    std::fs::write(
        options.output.join("run.txt"),
        format!(
            "paper=https://arxiv.org/abs/2201.02177v1\nstatus=running\nmodel=mesh_gpt\noperation={}\nmodulus={}\ntrain_fraction={}\ntraining_equations={}\nvalidation_equations={}\nseed={}\nabstract_symbol_permutation=true\nloss=answer_only\ncontext=4\nwidth={}\nheads={}\nlayers={}\nattention_mesh_depth={}\nffn_mesh_depth={}\nparameters={}\nall_parameters=mesh_edges\nattention=xsa\npositions=slice_rope\nnormalization=pre_rmsnorm\nqk_norm=per_head_rmsnorm\nrmsnorm_affine=false\nbias=false\nembeddings=mesh\nbetas=0.9,0.98\nepsilon=1e-8\nlearning_rate={}\nweight_decay={}\nwarmup_steps=10\nbatch={}\nsteps={}\nreport_every={}\nevaluation=complete_splits\ncheckpoint=weights_only\n",
            options.operation,
            options.modulus,
            options.train_fraction,
            dataset.split(false).len(),
            dataset.split(true).len(),
            options.seed,
            options.config.width,
            options.config.heads,
            options.config.layers,
            options.config.attention_mesh_depth,
            options.config.ffn_mesh_depth,
            options.config.parameter_count()?,
            options.optimizer.learning_rate,
            options.optimizer.weight_decay,
            batch,
            options.steps,
            options.report_every,
        ),
    )?;
    println!(
        "Mesh GPT grokking: {} mod {}, train={}, validation={}, batch={}, parameters={}",
        options.operation,
        options.modulus,
        dataset.split(false).len(),
        dataset.split(true).len(),
        batch,
        options.config.parameter_count()?
    );
    println!(
        "Answer-only loss; AdamW beta2=0.98, decay={}, warmup=10; continuing for {} updates",
        options.optimizer.weight_decay, options.steps
    );
    let mut history = BufWriter::new(File::create(options.output.join("loss.csv"))?);
    writeln!(
        history,
        "step,train_cross_entropy,validation_cross_entropy,train_accuracy,validation_accuracy,weight_l2,learning_rate,elapsed_seconds"
    )?;
    let start = Instant::now();
    let mut thresholds = Thresholds::default();
    record(
        &mut history,
        &model,
        &dataset,
        &options.output,
        0,
        0.0,
        start.elapsed().as_secs_f64(),
        &mut thresholds,
    )?;
    let mut trained_examples = 0;
    for step in 1..=options.steps {
        let (inputs, targets) = dataset.batch(false, sampler.sample(batch)?)?;
        let optimizer = warmed_optimizer(options.optimizer, step);
        model.train_batch_with_optimizer(&inputs, &targets, EQUATION_CONTEXT, optimizer)?;
        trained_examples += inputs.len() / EQUATION_CONTEXT;
        if step.is_multiple_of(options.report_every) || step == options.steps {
            record(
                &mut history,
                &model,
                &dataset,
                &options.output,
                step,
                optimizer.learning_rate,
                start.elapsed().as_secs_f64(),
                &mut thresholds,
            )?;
        }
    }
    write_embeddings(&options.output, &model, &dataset)?;
    write_predictions(&options.output, &model, &dataset)?;
    let metadata_path = options.output.join("run.txt");
    let metadata =
        std::fs::read_to_string(&metadata_path)?.replace("status=running", "status=complete");
    std::fs::write(&metadata_path, metadata)?;
    let mut metadata = OpenOptions::new().append(true).open(&metadata_path)?;
    writeln!(
        metadata,
        "elapsed_seconds={:.6}\ntraining_examples_seen={}\ncompleted_epochs={}\nexamples_in_partial_epoch={}\nfirst_recorded_train_99_step={}\nfirst_recorded_validation_99_step={}",
        start.elapsed().as_secs_f64(),
        trained_examples,
        trained_examples / dataset.split(false).len(),
        trained_examples % dataset.split(false).len(),
        observed(thresholds.training),
        observed(thresholds.validation)
    )?;
    println!(
        "Saved model, split, symbols, embeddings, predictions, and learning curves to {}",
        options.output.display()
    );
    println!(
        "First recorded accuracy >=99%: train={}, validation={}",
        observed(thresholds.training),
        observed(thresholds.validation)
    );
    Ok(())
}

fn warmed_optimizer(mut optimizer: AdamWOptions, step: usize) -> AdamWOptions {
    optimizer.learning_rate *= step.min(10) as f32 / 10.0;
    optimizer
}

fn observed(step: Option<usize>) -> String {
    step.map_or_else(|| "not_reached".into(), |step| step.to_string())
}

#[derive(Default)]
struct Thresholds {
    training: Option<usize>,
    validation: Option<usize>,
}

struct Metrics {
    loss: f64,
    accuracy: f64,
}

fn row_metrics(row: &[f32], target: usize) -> (f64, usize) {
    let mut prediction = 0;
    for token in 1..row.len() {
        if row[token] > row[prediction] {
            prediction = token;
        }
    }
    let maximum = f64::from(row[prediction]);
    let sum = row
        .iter()
        .map(|&x| (f64::from(x) - maximum).exp())
        .sum::<f64>();
    (sum.ln() + maximum - f64::from(row[target]), prediction)
}

fn evaluate(model: &GrokModel, dataset: &EquationDataset, validation: bool) -> Result<Metrics> {
    let count = dataset.split(validation).len();
    let (mut loss, mut correct) = (0.0, 0);
    for start in (0..count).step_by(512) {
        let indices: Vec<_> = (start..(start + 512).min(count)).collect();
        let (inputs, targets) = dataset.batch(validation, &indices)?;
        let logits = model.logits(&inputs, EQUATION_CONTEXT)?;
        for (x, y) in logits
            .as_chunks::<EQUATION_CONTEXT>()
            .0
            .iter()
            .zip(targets.as_chunks::<EQUATION_CONTEXT>().0.iter())
        {
            let (cross_entropy, prediction) = row_metrics(&x[3], y[3] as usize);
            loss += cross_entropy;
            correct += usize::from(prediction == y[3] as usize);
        }
    }
    Ok(Metrics {
        loss: loss / count as f64,
        accuracy: correct as f64 / count as f64,
    })
}

#[allow(clippy::too_many_arguments)]
fn record(
    history: &mut impl Write,
    model: &GrokModel,
    dataset: &EquationDataset,
    directory: &Path,
    step: usize,
    learning_rate: f32,
    elapsed: f64,
    thresholds: &mut Thresholds,
) -> Result<()> {
    let train = evaluate(model, dataset, false)?;
    let validation = evaluate(model, dataset, true)?;
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
    writeln!(
        history,
        "{step},{:.9},{:.9},{:.9},{:.9},{norm:.9},{learning_rate:.9},{elapsed:.6}",
        train.loss, validation.loss, train.accuracy, validation.accuracy
    )?;
    history.flush()?;
    Checkpoint {
        config: model.config(),
        tokenizer: dataset.tokenizer().clone(),
        parameters,
    }
    .save(directory.join("model.bin"))?;
    println!(
        "step {step:>7}: train CE={:.5} accuracy={:.2}%; validation CE={:.5} accuracy={:.2}%",
        train.loss,
        100.0 * train.accuracy,
        validation.loss,
        100.0 * validation.accuracy
    );
    Ok(())
}

fn write_dataset(directory: &Path, dataset: &EquationDataset) -> Result<()> {
    let mut equations = BufWriter::new(File::create(directory.join("equations.csv"))?);
    writeln!(
        equations,
        "split,pair_id,a,b,answer,input_a_id,operator_id,input_b_id,equals_id,target_id"
    )?;
    for validation in [false, true] {
        for equation in dataset.split(validation) {
            writeln!(
                equations,
                "{},{},{},{},{},{},{},{},{},{}",
                if validation { "validation" } else { "train" },
                equation.a * dataset.modulus() + equation.b,
                equation.a,
                equation.b,
                equation.answer,
                dataset.residue_tokens()[equation.a],
                dataset.modulus(),
                dataset.residue_tokens()[equation.b],
                dataset.modulus() + 1,
                dataset.residue_tokens()[equation.answer]
            )?;
        }
    }
    equations.flush()?;
    let mut symbols = BufWriter::new(File::create(directory.join("symbols.csv"))?);
    writeln!(symbols, "token_id,symbol")?;
    for token in 0..dataset.tokenizer().vocab_size() {
        writeln!(symbols, "{token},{}", dataset.symbol_label(token)?)?;
    }
    symbols.flush()?;
    Ok(())
}

fn write_embeddings(directory: &Path, model: &GrokModel, dataset: &EquationDataset) -> Result<()> {
    let mut file = BufWriter::new(File::create(directory.join("embeddings.csv"))?);
    write!(file, "token_id,symbol")?;
    for channel in 0..model.config().width {
        write!(file, ",c{channel}")?;
    }
    writeln!(file)?;
    for (token, row) in model.token_embeddings()?.iter().enumerate() {
        write!(file, "{token},{}", dataset.symbol_label(token)?)?;
        for value in row {
            write!(file, ",{value:.9}")?;
        }
        writeln!(file)?;
    }
    file.flush()?;
    Ok(())
}

fn write_predictions(directory: &Path, model: &GrokModel, dataset: &EquationDataset) -> Result<()> {
    let mut file = BufWriter::new(File::create(directory.join("predictions.csv"))?);
    writeln!(file, "split,a,b,answer,prediction,correct,cross_entropy")?;
    for validation in [false, true] {
        let split = dataset.split(validation);
        for start in (0..split.len()).step_by(512) {
            let indices: Vec<_> = (start..(start + 512).min(split.len())).collect();
            let (inputs, targets) = dataset.batch(validation, &indices)?;
            let logits = model.logits(&inputs, EQUATION_CONTEXT)?;
            for ((&index, x), y) in indices
                .iter()
                .zip(logits.as_chunks::<4>().0.iter())
                .zip(targets.as_chunks::<4>().0.iter())
            {
                let (loss, prediction) = row_metrics(&x[3], y[3] as usize);
                let equation = split[index];
                writeln!(
                    file,
                    "{},{},{},{},{},{},{loss:.9}",
                    if validation { "validation" } else { "train" },
                    equation.a,
                    equation.b,
                    equation.answer,
                    dataset.symbol_label(prediction)?,
                    usize::from(prediction == y[3] as usize)
                )?;
            }
        }
    }
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grokking_defaults_and_warmup_follow_the_paper_protocol() {
        let options = Options::parse(std::iter::empty()).unwrap();
        assert_eq!(options.modulus, 97);
        assert_eq!(options.train_fraction, 0.5);
        assert_eq!(options.steps, 100_000);
        assert_eq!(options.config.width, 128);
        assert_eq!(options.config.layers, 2);
        assert_eq!(options.config.heads, 4);
        assert_eq!(options.optimizer.beta2, 0.98);
        assert_eq!(options.optimizer.weight_decay, 1.0);
        for step in [1, 5, 10, 11, 100_000] {
            let actual = warmed_optimizer(options.optimizer, step);
            assert!((actual.learning_rate - 0.001 * step.min(10) as f32 / 10.0).abs() < 1e-9);
        }
        let (loss, prediction) = row_metrics(&[1000.0, 1001.0, 999.0], 1);
        assert_eq!(prediction, 1);
        assert!((loss - (1.0 + (-1.0_f64).exp() + (-2.0_f64).exp()).ln()).abs() < 1e-12);
    }
}
