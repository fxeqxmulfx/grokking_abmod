//! Repeated, warmed end-to-end training timings for the mod-97 Mesh GPT workload.

use std::{fs::File, io::Write, path::PathBuf, str::FromStr, time::Instant};

use grokking_abmod::{
    Result,
    gpt::{
        AdamWOptions, Checkpoint, EQUATION_CONTEXT, EquationDataset, GptConfig, GpuMeshGpt,
        ModularOperation,
    },
};

fn value<T: FromStr>(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<T> {
    arguments
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))?
        .parse()
        .map_err(|_| format!("invalid value for {flag}").into())
}

fn main() -> Result<()> {
    let (mut steps, mut warmup, mut trials, mut batch, mut device) = (100, 10, 3, 512, 0);
    let mut directory = PathBuf::from("target/mesh-gpt-bench");
    let mut config = GptConfig {
        width: 128,
        context: EQUATION_CONTEXT,
        ..GptConfig::default()
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--steps" => steps = value(&mut arguments, &flag)?,
            "--warmup" => warmup = value(&mut arguments, &flag)?,
            "--trials" => trials = value(&mut arguments, &flag)?,
            "--batch" => batch = value(&mut arguments, &flag)?,
            "--width" => config.width = value(&mut arguments, &flag)?,
            "--heads" => config.heads = value(&mut arguments, &flag)?,
            "--device" => device = value(&mut arguments, &flag)?,
            "--output" => directory = value(&mut arguments, &flag)?,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo oxide run --bin mesh-gpt-bench -- [options]\n\
                    --steps N     Timed updates per trial (default: 100)\n\
                    --warmup N    Untimed updates before each trial (default: 10)\n\
                    --trials N    Repetitions from identical weights (default: 3)\n\
                    --batch N     Fixed training equations (default: 512)\n\
                    --width N     Channels (default: 128)\n\
                    --heads N     Attention heads (default: 4)\n\
                    --device N    CUDA device ordinal (default: 0)\n\
                    --output PATH Artifacts (default: target/mesh-gpt-bench)"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown benchmark option: {flag}").into()),
        }
    }
    if steps == 0 || trials == 0 || batch == 0 || batch > 1024 {
        return Err("steps/trials must be positive and batch must be in 1..1024".into());
    }
    let dataset = EquationDataset::new(97, ModularOperation::Add, 0.5, 7)?;
    let indices: Vec<_> = (0..batch).collect();
    let (inputs, targets) = dataset.batch(false, &indices)?;
    config.vocab_size = dataset.tokenizer().vocab_size();
    config.validate()?;
    let optimizer = AdamWOptions {
        learning_rate: 0.001,
        weight_decay: 1.0,
        beta2: 0.98,
        ..AdamWOptions::default()
    };
    let mut model = GpuMeshGpt::new(device, config, 7)?;
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
            "precision=fp32\nmodel_type={}\nworkload=add_mod97\nseed=7\nwidth={}\nheads={}\nlayers={}\nattention_mesh_depth={}\nffn_mesh_depth={}\nvocabulary={}\ncontext=4\nbatch={}\nsteps={}\nwarmup={}\ntrials={}\nlearning_rate=0.001\nweight_decay=1\nbetas=0.9,0.98\ntiming=end_to_end_wall_clock\nfinal_parameter_download=included\nvalidation_and_artifact_io=excluded\n",
            std::any::type_name::<GpuMeshGpt>(),
            config.width,
            config.heads,
            config.layers,
            config.attention_mesh_depth,
            config.ffn_mesh_depth,
            config.vocab_size,
            batch,
            steps,
            warmup,
            trials
        ),
    )?;
    for trial in 1..=trials {
        model.set_parameters(&initial)?;
        for _ in 0..warmup {
            model.train_batch_with_optimizer(&inputs, &targets, EQUATION_CONTEXT, optimizer)?;
        }
        // A parameter download waits for the warmup's final AdamW update.
        model.parameters()?;
        let start = Instant::now();
        let (mut first_loss, mut last_loss) = (0.0, 0.0);
        for step in 0..steps {
            last_loss =
                model.train_batch_with_optimizer(&inputs, &targets, EQUATION_CONTEXT, optimizer)?;
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
        println!("trial {trial}: {ms_per_step:.3} ms/update; CE {first_loss:.5} -> {last_loss:.5}");
        if trial == 1 {
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
