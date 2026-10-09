use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    str::FromStr,
    time::Instant,
};

use grokking_abmod::{
    Result,
    gpt::{
        AdamWOptions, CharacterTokenizer, Checkpoint, GptConfig, GptTrainOptions, GpuMeshGpt,
        IGNORE_TARGET, ReferenceGpt, SpecializedMeshGpt, TextDataset,
    },
};

#[path = "mesh-gpt/grokking.rs"]
mod grokking;

struct Options {
    command: String,
    config: GptConfig,
    training: GptTrainOptions,
    device: usize,
    seed: u64,
    batch: usize,
    dataset: PathBuf,
    output: PathBuf,
    checkpoint: Option<PathBuf>,
    prompt: String,
    tokens: usize,
    temperature: f32,
    help: bool,
}

impl Options {
    fn parse() -> Result<Self> {
        let mut options = Self {
            command: "train".into(),
            config: GptConfig::default(),
            training: GptTrainOptions::default(),
            device: 0,
            seed: 7,
            batch: 8,
            dataset: PathBuf::from("data/mesh-gpt-demo.txt"),
            output: PathBuf::from("target/mesh-gpt"),
            checkpoint: None,
            prompt: "the little robot ".into(),
            tokens: 240,
            temperature: 0.7,
            help: false,
        };
        let mut arguments = std::env::args().skip(1).peekable();
        if arguments.peek().is_some_and(|a| !a.starts_with('-')) {
            options.command = arguments.next().unwrap();
        }
        while let Some(flag) = arguments.next() {
            match flag.as_str() {
                "--width" => options.config.width = value(&mut arguments, &flag)?,
                "--heads" => options.config.heads = value(&mut arguments, &flag)?,
                "--layers" => options.config.layers = value(&mut arguments, &flag)?,
                "--context" => options.config.context = value(&mut arguments, &flag)?,
                "--attention-depth" => {
                    options.config.attention_mesh_depth = value(&mut arguments, &flag)?
                }
                "--ffn-depth" => options.config.ffn_mesh_depth = value(&mut arguments, &flag)?,
                "--steps" => options.training.steps = value(&mut arguments, &flag)?,
                "--learning-rate" => options.training.learning_rate = value(&mut arguments, &flag)?,
                "--weight-decay" => options.training.weight_decay = value(&mut arguments, &flag)?,
                "--report-every" => options.training.report_every = value(&mut arguments, &flag)?,
                "--batch" => options.batch = value(&mut arguments, &flag)?,
                "--seed" => options.seed = value(&mut arguments, &flag)?,
                "--device" => options.device = value(&mut arguments, &flag)?,
                "--dataset" => options.dataset = value(&mut arguments, &flag)?,
                "--output" => options.output = value(&mut arguments, &flag)?,
                "--checkpoint" => options.checkpoint = Some(value(&mut arguments, &flag)?),
                "--prompt" => options.prompt = value(&mut arguments, &flag)?,
                "--tokens" => options.tokens = value(&mut arguments, &flag)?,
                "--temperature" => options.temperature = value(&mut arguments, &flag)?,
                "--help" | "-h" => options.help = true,
                _ => return Err(format!("unknown option: {flag}; use --help").into()),
            }
        }
        Ok(options)
    }
}

fn value<T: FromStr>(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<T> {
    let text = arguments
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))?;
    text.parse()
        .map_err(|_| format!("invalid value for {flag}: {text}").into())
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("grok") {
        return grokking::run(std::env::args().skip(2));
    }
    let options = Options::parse()?;
    if options.help {
        println!(
            "Mesh GPT: all learned weights in meshes, with fixed RMSNorm, QK-Norm, slice RoPE, and XSA\n\
            Usage: cargo oxide run --bin mesh-gpt -- [train|generate|grok|verify] [options]\n\n\
            grok --help           Modular arithmetic experiments from arXiv:2201.02177v1\n\
            --dataset PATH        UTF-8 training corpus (default: data/mesh-gpt-demo.txt)\n\
            --output PATH         Artifacts (default: target/mesh-gpt)\n\
            --checkpoint PATH     Load for generate; save for train (default: OUTPUT/model.bin)\n\
            --width N             Channels, 2..128 (default: 32)\n\
            --heads N             Heads, 1..8, divides width, even channels/head (default: 4)\n\
            --layers N            Transformer blocks, 1..64 (default: 2)\n\
            --context N           Maximum sequence length, 1..128 (default: 32)\n\
            --attention-depth N   Embedding/Q/K/V/output mesh depth, 1..256 (default: 4)\n\
            --ffn-depth N         Feed-forward mesh depth, 1..256 (default: 6)\n\
            --steps N             AdamW steps (default: 1000)\n\
            --batch N             Sequences per batch (default: 8, at most 4096 token rows)\n\
            --learning-rate X     AdamW learning rate (default: 0.003)\n\
            --weight-decay X      AdamW decay on all parameters (default: 0.01)\n\
            --report-every N      Report interval (default: 100)\n\
            --prompt TEXT         Generation prefix (default: 'the little robot ')\n\
            --tokens N            New characters (default: 240)\n\
            --temperature X       Sampling temperature, 0 is greedy (default: 0.7)\n\
            --seed N              Initialization/data/sampling seed (default: 7)\n\
            --device N            CUDA device ordinal (default: 0)"
        );
        return Ok(());
    }
    if options.command == "verify" {
        return verify(options.device);
    }
    if !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err("temperature must be finite and nonnegative".into());
    }
    let checkpoint = options
        .checkpoint
        .clone()
        .unwrap_or_else(|| options.output.join("model.bin"));
    match options.command.as_str() {
        "generate" => {
            let saved = Checkpoint::load(checkpoint)?;
            let mut model = GpuMeshGpt::new(options.device, saved.config, options.seed)?;
            model.set_parameters(&saved.parameters)?;
            println!(
                "{}",
                model.generate(
                    &saved.tokenizer,
                    &options.prompt,
                    options.tokens,
                    options.temperature,
                    options.seed
                )?
            );
            Ok(())
        }
        "train" => train(options, checkpoint),
        other => {
            Err(format!("unknown command: {other}; use train, generate, grok, or verify").into())
        }
    }
}

fn train(mut options: Options, checkpoint: PathBuf) -> Result<()> {
    let text = std::fs::read_to_string(&options.dataset)?;
    let tokenizer = CharacterTokenizer::from_text(&text)?;
    options.config.vocab_size = tokenizer.vocab_size();
    options.config.validate()?;
    options.training.validate()?;
    if options.prompt.is_empty() {
        return Err("generation prompt must not be empty".into());
    }
    tokenizer.encode(&options.prompt)?;
    let dataset = TextDataset::new(tokenizer.encode(&text)?, options.config)?;
    let (evaluation_x, evaluation_y) =
        dataset.batch(true, options.config.context, options.batch, 101)?;
    let mut model = GpuMeshGpt::new(options.device, options.config, options.seed)?;
    println!(
        "Mesh GPT (bias-free, fixed RMSNorm + QK-Norm, slice RoPE + XSA): blocks={}, width={}, heads={}, context={}, vocabulary={}, parameters={}",
        options.config.layers,
        options.config.width,
        options.config.heads,
        options.config.context,
        tokenizer.vocab_size(),
        options.config.parameter_count()?
    );
    println!(
        "Meshes: token embedding/Q/K/V/output depth={}, FFN depth={}; all weights are mesh edges",
        options.config.attention_mesh_depth, options.config.ffn_mesh_depth
    );
    std::fs::create_dir_all(&options.output)?;
    let mut history = BufWriter::new(File::create(options.output.join("loss.csv"))?);
    writeln!(
        history,
        "step,train_cross_entropy,validation_cross_entropy,validation_perplexity"
    )?;
    let (initial_x, initial_y) =
        dataset.batch(false, options.config.context, options.batch, options.seed)?;
    let initial_train = model.cross_entropy(&initial_x, &initial_y, options.config.context)?;
    let initial_validation =
        model.cross_entropy(&evaluation_x, &evaluation_y, options.config.context)?;
    report(&mut history, 0, initial_train, initial_validation)?;
    let start = Instant::now();
    let mut sum = 0.0;
    let mut count = 0;
    for step in 1..=options.training.steps {
        let (x, y) = dataset.batch(
            false,
            options.config.context,
            options.batch,
            options.seed.wrapping_add(step as u64),
        )?;
        sum += model.train_batch(
            &x,
            &y,
            options.config.context,
            options.training.learning_rate,
            options.training.weight_decay,
        )?;
        count += 1;
        if step.is_multiple_of(options.training.report_every) || step == options.training.steps {
            let validation =
                model.cross_entropy(&evaluation_x, &evaluation_y, options.config.context)?;
            report(&mut history, step, sum / count as f64, validation)?;
            sum = 0.0;
            count = 0;
        }
    }
    history.flush()?;
    let elapsed = start.elapsed().as_secs_f64();
    println!("Training time: {elapsed:.3} s");
    let parameters = model.parameters()?;
    if let Some(parent) = checkpoint.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Checkpoint {
        config: options.config,
        tokenizer: tokenizer.clone(),
        parameters,
    }
    .save(&checkpoint)?;
    let generated = model.generate(
        &tokenizer,
        &options.prompt,
        options.tokens,
        options.temperature,
        options.seed,
    )?;
    std::fs::write(options.output.join("sample.txt"), &generated)?;
    let final_validation =
        model.cross_entropy(&evaluation_x, &evaluation_y, options.config.context)?;
    std::fs::write(
        options.output.join("run.txt"),
        format!(
            "attention=xsa\npositions=slice_rope\nrope_base=10000\nnormalization=pre_rmsnorm\nrmsnorm_epsilon=1e-5\nrmsnorm_affine=false\nqk_norm=per_head_rmsnorm\nbias=false\nembeddings=mesh\ntoken_codes=signed_binary_parity_radix\nembedding_depth=attention_mesh_depth\nhead=tied_mesh_embeddings\nall_parameters=mesh_edges\ndataset={}\nseed={}\nsteps={}\nbatch={}\nwidth={}\nheads={}\nlayers={}\ncontext={}\nattention_mesh_depth={}\nffn_mesh_depth={}\nparameters={}\nlearning_rate={}\nweight_decay={}\ntraining_seconds={elapsed:.6}\ninitial_validation_cross_entropy={initial_validation:.9}\nfinal_validation_cross_entropy={final_validation:.9}\n",
            options.dataset.display(),
            options.seed,
            options.training.steps,
            options.batch,
            options.config.width,
            options.config.heads,
            options.config.layers,
            options.config.context,
            options.config.attention_mesh_depth,
            options.config.ffn_mesh_depth,
            options.config.parameter_count()?,
            options.training.learning_rate,
            options.training.weight_decay,
        ),
    )?;
    println!(
        "Saved checkpoint to {}\n\n{generated}",
        checkpoint.display()
    );
    Ok(())
}

fn report(writer: &mut impl Write, step: usize, train: f64, validation: f64) -> Result<()> {
    println!(
        "step {step:>6}: train CE={train:.5}, validation CE={validation:.5}, perplexity={:.3}",
        validation.exp()
    );
    writeln!(
        writer,
        "{step},{train:.9},{validation:.9},{:.9}",
        validation.exp()
    )?;
    Ok(())
}

fn close(name: &str, expected: f64, actual: f64) -> Result<()> {
    if !expected.is_finite()
        || !actual.is_finite()
        || (expected - actual).abs() > 5e-4 * (1.0 + expected.abs())
    {
        return Err(format!("{name}: expected {expected:.9}, got {actual:.9}").into());
    }
    Ok(())
}

fn verify_model<const W: usize, const V: usize, const C: usize, const H: usize>(
    device: usize,
    config: GptConfig,
    cases: &[(usize, usize)],
) -> Result<()> {
    let reference = ReferenceGpt::new(config, 7)?;
    let mut model = SpecializedMeshGpt::<W, V, C, H>::new(device, config, 7)?;
    for &(batch, sequence) in cases {
        let x: Vec<_> = (0..batch * sequence)
            .map(|i| (i % config.vocab_size) as u32)
            .collect();
        let y: Vec<_> = (0..x.len())
            .map(|i| ((i + 2) % config.vocab_size) as u32)
            .collect();
        let expected = reference.logits(&x, sequence)?;
        let actual = model.logits(&x, sequence)?;
        for (row, (cpu, gpu)) in expected.iter().zip(&actual).enumerate() {
            for (token, (&cpu, &gpu)) in cpu.iter().zip(gpu).enumerate() {
                close(
                    &format!("logit row={row} token={token}"),
                    f64::from(cpu),
                    f64::from(gpu),
                )?;
            }
        }
        let answer_only: Vec<_> = y
            .iter()
            .enumerate()
            .map(|(row, &target)| {
                if row % sequence + 1 == sequence {
                    target
                } else {
                    IGNORE_TARGET
                }
            })
            .collect();
        for targets in [&y, &answer_only] {
            let (cpu_loss, cpu_gradients) = reference.loss_and_gradient(&x, targets, sequence)?;
            let (gpu_loss, gpu_gradients) = model.loss_and_gradient(&x, targets, sequence)?;
            close("cross-entropy", cpu_loss, gpu_loss)?;
            for (parameter, (&cpu, &gpu)) in cpu_gradients.iter().zip(&gpu_gradients).enumerate() {
                close(&format!("gradient parameter={parameter}"), cpu, gpu)?;
            }
        }
    }
    let prefix = [0, 1];
    let mut tokens = vec![0, 1, 2];
    let full = model.logits(&tokens, 3)?;
    tokens[2] = 4;
    let changed = model.logits(&tokens, 3)?;
    if full[..2] != changed[..2] || full[..2] != model.logits(&prefix, 2)? {
        return Err("causal mask leaked future tokens".into());
    }
    if model.logits(&[], 1).is_ok()
        || model.logits(&[u32::MAX], 1).is_ok()
        || model.logits(&[0], 0).is_ok()
        || model.loss_and_gradient(&[0], &[u32::MAX], 1).is_ok()
        || model.set_parameters(&[0.0]).is_ok()
    {
        return Err("GPT validation failed".into());
    }

    let x = [0, 1, 3, 2, 4, 1];
    for optimizer in [
        AdamWOptions {
            learning_rate: 0.002,
            weight_decay: 0.01,
            ..AdamWOptions::default()
        },
        AdamWOptions {
            learning_rate: 0.001,
            weight_decay: 1.0,
            beta2: 0.98,
            ..AdamWOptions::default()
        },
    ] {
        let y = if optimizer.beta2 == 0.98 {
            [
                IGNORE_TARGET,
                IGNORE_TARGET,
                2,
                IGNORE_TARGET,
                IGNORE_TARGET,
                0,
            ]
        } else {
            [1, 3, 2, 4, 1, 0]
        };
        model.set_parameters(reference.parameters())?;
        let mut parameters = reference.parameters().to_vec();
        let mut moments = vec![[0.0_f32; 2]; parameters.len()];
        let (mut b1, mut b2) = (1.0_f32, 1.0_f32);
        let mut first_step = Vec::new();
        for step in 1..=3 {
            let cpu = ReferenceGpt::from_parameters(config, parameters.clone())?;
            let (_, gradient) = cpu.loss_and_gradient(&x, &y, 3)?;
            b1 *= optimizer.beta1;
            b2 *= optimizer.beta2;
            for ((parameter, moment), gradient) in
                parameters.iter_mut().zip(&mut moments).zip(gradient)
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
            if step == 1 {
                first_step = parameters.clone();
            }
            model.train_batch_with_optimizer(&x, &y, 3, optimizer)?;
            for (parameter, (&cpu, gpu)) in parameters.iter().zip(model.parameters()?).enumerate() {
                close(
                    &format!("AdamW step={step} parameter={parameter}"),
                    f64::from(cpu),
                    f64::from(gpu),
                )?;
            }
        }
        model.set_parameters(reference.parameters())?;
        model.train_batch_with_optimizer(&x, &y, 3, optimizer)?;
        for (parameter, (&cpu, gpu)) in first_step.iter().zip(model.parameters()?).enumerate() {
            close(
                &format!("AdamW reset parameter={parameter}"),
                f64::from(cpu),
                f64::from(gpu),
            )?;
        }
    }
    let mut zero_embeddings = vec![0.5; config.parameter_count()?];
    zero_embeddings[..2 * config.width * config.attention_mesh_depth].fill(0.0);
    model.set_parameters(&zero_embeddings)?;
    if model.logits(&x, 3)?.iter().flatten().any(|&x| x != 0.0) {
        return Err("zero token embeddings produced additive logit offsets".into());
    }
    println!(
        "PASS mesh embeddings/RMSNorm/QK-Norm/RoPE/XSA width={} heads={} layers={}: logits, dense/masked gradients, causality, batch tails, AdamW beta2=0.999/0.98/reset, zero-input offsets",
        config.width, config.heads, config.layers
    );
    for oversized in [
        GptConfig {
            width: W * 2,
            ..config
        },
        GptConfig {
            vocab_size: V + 1,
            ..config
        },
        GptConfig {
            context: C + 1,
            ..config
        },
        GptConfig {
            width: 2 * (H + 1),
            heads: H + 1,
            ..config
        },
    ] {
        if SpecializedMeshGpt::<W, V, C, H>::new(device, oversized, 7).is_ok() {
            return Err("model accepted dimensions exceeding its kernel capacities".into());
        }
    }
    Ok(())
}

fn verify(device: usize) -> Result<()> {
    GpuMeshGpt::verify_embedding_gradient_reduction(device)?;
    println!(
        "PASS isolated tied-embedding gradients: f64 sums, input/head contributions, chunk tails, up to 4096 rows, untouched padding, contracts"
    );
    GpuMeshGpt::verify_mesh_gradient_reduction(device)?;
    println!(
        "PASS isolated mesh gradient reductions: f64 sums, chunk boundaries/tails, up to 4096 rows, untouched layers, contracts"
    );
    GpuMeshGpt::verify_rmsnorm(device)?;
    println!(
        "PASS isolated RMSNorm/QK-Norm: outputs, input gradients, per-head groups, zero/tiny values, batch tails, contracts"
    );
    GpuMeshGpt::verify_rotary_embeddings(device)?;
    println!(
        "PASS isolated slice RoPE: split-half pairs, norms, inverse/backward, positions, batch tails, contracts"
    );
    GpuMeshGpt::verify_exclusive_attention(device)?;
    println!(
        "PASS isolated XSA: per-head orthogonality, both derivatives, zero/tiny values, batch tails"
    );
    for config in [
        GptConfig {
            vocab_size: 129,
            context: 3,
            width: 2,
            heads: 1,
            layers: 1,
            attention_mesh_depth: 2,
            ffn_mesh_depth: 2,
        },
        GptConfig {
            vocab_size: 5,
            context: 4,
            width: 4,
            heads: 2,
            layers: 1,
            attention_mesh_depth: 2,
            ffn_mesh_depth: 2,
        },
        GptConfig {
            vocab_size: 7,
            context: 5,
            width: 8,
            heads: 2,
            layers: 2,
            attention_mesh_depth: 3,
            ffn_mesh_depth: 4,
        },
        GptConfig {
            vocab_size: 9,
            context: 3,
            width: 64,
            heads: 8,
            layers: 1,
            attention_mesh_depth: 1,
            ffn_mesh_depth: 1,
        },
        GptConfig {
            vocab_size: 9,
            context: 3,
            width: 128,
            heads: 4,
            layers: 2,
            attention_mesh_depth: 2,
            ffn_mesh_depth: 2,
        },
    ] {
        verify_model::<128, 256, 128, 8>(device, config, &[(1, 1), (1, 3), (2, 3), (43, 3)])?;
    }
    verify_model::<4, 8, 8, 2>(
        device,
        GptConfig {
            vocab_size: 5,
            context: 5,
            width: 4,
            heads: 2,
            layers: 3,
            attention_mesh_depth: 2,
            ffn_mesh_depth: 2,
        },
        &[(1, 1), (2, 3), (43, 3)],
    )?;
    verify_model::<6, 13, 7, 3>(
        device,
        GptConfig {
            vocab_size: 11,
            context: 5,
            width: 6,
            heads: 3,
            layers: 1,
            attention_mesh_depth: 2,
            ffn_mesh_depth: 2,
        },
        &[(1, 1), (2, 3), (43, 3)],
    )?;
    verify_model::<256, 512, 256, 16>(
        device,
        GptConfig {
            vocab_size: 257,
            context: 129,
            width: 256,
            heads: 16,
            layers: 3,
            attention_mesh_depth: 2,
            ffn_mesh_depth: 2,
        },
        &[(1, 1), (1, 3), (2, 3)],
    )?;
    verify_model::<4, 8, 256, 2>(
        device,
        GptConfig {
            vocab_size: 5,
            context: 129,
            width: 4,
            heads: 2,
            layers: 1,
            attention_mesh_depth: 1,
            ffn_mesh_depth: 1,
        },
        &[(1, 129)],
    )?;
    // The identical final input token needs different predictions depending on
    // the first token. A token-local model cannot solve both examples.
    let config = GptConfig {
        vocab_size: 5,
        context: 3,
        width: 8,
        heads: 2,
        layers: 1,
        attention_mesh_depth: 2,
        ffn_mesh_depth: 2,
    };
    let mut model = GpuMeshGpt::new(device, config, 11)?;
    let inputs = [0, 1, 2, 4, 1, 2];
    let targets = [1, 2, 3, 1, 2, 0];
    let initial = model.cross_entropy(&inputs, &targets, 3)?;
    for _ in 0..400 {
        model.train_batch(&inputs, &targets, 3, 0.01, 0.0)?;
    }
    let loss = model.cross_entropy(&inputs, &targets, 3)?;
    let logits = model.logits(&inputs, 3)?;
    for (&target, row) in targets.iter().zip(logits) {
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or("empty logits")?
            .0;
        if best != target as usize {
            return Err("mesh Transformer did not learn context-dependent prediction".into());
        }
    }
    if loss >= 0.1 || loss >= initial / 10.0 {
        return Err(format!("context task loss did not converge: {initial} -> {loss}").into());
    }
    println!("PASS context-dependent learning: CE {initial:.5} -> {loss:.5}");
    println!("CUDA Mesh GPT verification passed");
    Ok(())
}
