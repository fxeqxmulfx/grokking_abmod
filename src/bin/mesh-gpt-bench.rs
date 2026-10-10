//! Repeated FP32 training timings for equations and long-context token sequences.

#[path = "../cli.rs"]
mod cli;

use cli::value;

use std::{fs::File, io::Write, path::PathBuf, time::Instant};

use grokking_abmod::{
    Result,
    gpt::{
        AdamWOptions, Checkpoint, EQUATION_CONTEXT, EquationDataset, GptConfig, MAX_ROWS,
        ModularOperation, SpecializedMeshGpt,
    },
};

struct Options {
    config: GptConfig,
    steps: usize,
    warmup: usize,
    trials: usize,
    batch: usize,
    device: usize,
    directory: PathBuf,
}

fn main() -> Result<()> {
    let mut options = Options {
        config: GptConfig {
            width: 128,
            context: EQUATION_CONTEXT,
            ..GptConfig::default()
        },
        steps: 100,
        warmup: 10,
        trials: 3,
        batch: 512,
        device: 0,
        directory: PathBuf::from("target/mesh-gpt-bench"),
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--steps" => options.steps = value(&mut arguments, &flag)?,
            "--warmup" => options.warmup = value(&mut arguments, &flag)?,
            "--trials" => options.trials = value(&mut arguments, &flag)?,
            "--batch" => options.batch = value(&mut arguments, &flag)?,
            "--width" => options.config.width = value(&mut arguments, &flag)?,
            "--heads" => options.config.heads = value(&mut arguments, &flag)?,
            "--context" => options.config.context = value(&mut arguments, &flag)?,
            "--layers" => options.config.layers = value(&mut arguments, &flag)?,
            "--attention-depth" => {
                options.config.attention_mesh_depth = value(&mut arguments, &flag)?
            }
            "--ffn-depth" => options.config.ffn_mesh_depth = value(&mut arguments, &flag)?,
            "--device" => options.device = value(&mut arguments, &flag)?,
            "--output" => options.directory = value(&mut arguments, &flag)?,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo oxide run --bin mesh-gpt-bench -- [options]\n\
                    --steps N     Timed updates per trial (default: 100)\n\
                    --warmup N    Untimed updates before each trial (default: 10)\n\
                    --trials N    Repetitions from identical weights (default: 3)\n\
                    --batch N     Fixed sequences (default: 512)\n\
                    --width N     Channels (default: 128)\n\
                    --heads N     Attention heads (default: 4)\n\
                    --context N   Tokens per sequence (default: 4)\n\
                    --layers N    Decoder blocks (default: 2)\n\
                    --attention-depth N  Mesh layers per attention projection (default: 4)\n\
                    --ffn-depth N  Mesh layers per feed-forward projection (default: 6)\n\
                    --device N    CUDA device ordinal (default: 0)\n\
                    --output PATH Artifacts (default: target/mesh-gpt-bench)"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown benchmark option: {flag}").into()),
        }
    }
    options.config.validate()?;
    if options.steps == 0
        || options.trials == 0
        || options.batch == 0
        || options
            .batch
            .checked_mul(options.config.context)
            .is_none_or(|rows| rows > MAX_ROWS)
    {
        return Err(format!(
            "steps/trials/batch must be positive, with at most {MAX_ROWS} token rows"
        )
        .into());
    }
    match (options.config.width, options.config.context, options.config.heads) {
        (..=128, ..=128, ..=8) => benchmark::<128, 128>(options),
        (..=256, ..=128, ..=8) => benchmark::<256, 128>(options),
        (..=512, ..=128, ..=8) => benchmark::<512, 128>(options),
        (..=1024, ..=128, ..=8) => benchmark::<1024, 128>(options),
        (..=128, ..=1024, ..=8) => benchmark::<128, 1024>(options),
        (..=128, ..=4096, ..=8) => benchmark::<128, 4096>(options),
        _ => Err("benchmark supports width <=1024 at context <=128, or width <=128 at context <=4096, with <=8 heads".into()),
    }
}

fn benchmark<const W: usize, const C: usize>(options: Options) -> Result<()> {
    let Options {
        mut config,
        steps,
        warmup,
        trials,
        batch,
        device,
        directory,
    } = options;
    let dataset = EquationDataset::new(97, ModularOperation::Add, 0.5, 7)?;
    let (inputs, targets, workload) = if config.context == EQUATION_CONTEXT {
        let indices: Vec<_> = (0..batch).map(|i| i % dataset.split(false).len()).collect();
        let (inputs, targets) = dataset.batch(false, &indices)?;
        (inputs, targets, "add_mod97")
    } else {
        let mut inputs = Vec::with_capacity(batch * config.context);
        let mut targets = Vec::with_capacity(inputs.capacity());
        for sequence in 0..batch {
            let start = (7 + sequence * 37) % dataset.tokenizer().vocab_size();
            for position in 0..config.context {
                inputs.push(((start + position) % dataset.tokenizer().vocab_size()) as u32);
                targets.push(((start + position + 1) % dataset.tokenizer().vocab_size()) as u32);
            }
        }
        (inputs, targets, "cyclic_next_token")
    };
    config.vocab_size = dataset.tokenizer().vocab_size();
    config.validate()?;
    let optimizer = AdamWOptions {
        learning_rate: 0.001,
        weight_decay: 1.0,
        beta2: 0.98,
        ..AdamWOptions::default()
    };
    let mut model = SpecializedMeshGpt::<W, 256, C, 8>::new(device, config, 7)?;
    let initial = model.parameters()?;
    std::fs::create_dir_all(&directory)?;
    let mut file = File::create(directory.join("timings.csv"))?;
    writeln!(
        file,
        "trial,steps,total_ms,ms_per_step,first_loss,last_loss,weight_l2"
    )?;
    std::fs::write(
        directory.join("run.txt"),
        format!(
            "precision=fp32\nmodel_type={}\nworkload={}\nseed=7\nwidth={}\nheads={}\nlayers={}\nattention_mesh_depth={}\nffn_mesh_depth={}\nvocabulary={}\ncontext={}\nbatch={}\nsteps={}\nwarmup={}\ntrials={}\nlearning_rate=0.001\nweight_decay=1\nbetas=0.9,0.98\ntiming=end_to_end_wall_clock\nfinal_parameter_download=included\nvalidation_and_artifact_io=excluded\n",
            std::any::type_name::<SpecializedMeshGpt<W, 256, C, 8>>(),
            workload,
            config.width,
            config.heads,
            config.layers,
            config.attention_mesh_depth,
            config.ffn_mesh_depth,
            config.vocab_size,
            config.context,
            batch,
            steps,
            warmup,
            trials
        ),
    )?;
    for trial in 1..=trials {
        model.set_parameters(&initial)?;
        for _ in 0..warmup {
            model.train_batch_with_optimizer(&inputs, &targets, config.context, optimizer)?;
        }
        // A parameter download waits for the warmup's final AdamW update.
        model.parameters()?;
        let start = Instant::now();
        let (mut first_loss, mut last_loss) = (0.0, 0.0);
        for step in 0..steps {
            last_loss =
                model.train_batch_with_optimizer(&inputs, &targets, config.context, optimizer)?;
            if step == 0 {
                first_loss = last_loss;
            }
        }
        // Include completion of the last update, not just its host launch time.
        let parameters = model.parameters()?;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let ms_per_step = total_ms / steps as f64;
        let norm = parameters
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        writeln!(
            file,
            "{trial},{steps},{total_ms:.6},{ms_per_step:.6},{first_loss:.9},{last_loss:.9},{norm:.9}"
        )?;
        println!(
            "trial {trial}: {ms_per_step:.3} ms/update; {:.0} tokens/s; CE {first_loss:.5} -> {last_loss:.5}",
            inputs.len() as f64 * 1000.0 / ms_per_step
        );
        if trial == 1 {
            let bytes = model.allocated_device_bytes();
            println!("device buffers: {:.1} MiB", bytes as f64 / 1048576.0);
            writeln!(
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(directory.join("run.txt"))?,
                "allocated_device_bytes={bytes}"
            )?;
            Checkpoint {
                config,
                tokenizer: dataset.tokenizer().clone(),
                parameters,
            }
            .save(directory.join("model.bin"))?;
        }
    }
    Ok(())
}
