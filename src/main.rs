use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    str::FromStr,
};

use grokking_abmod::{
    GpuMesh, MeshTopology, ReferenceMesh, Result, TrainOptions, piecewise_target,
};

struct Options {
    depth: usize,
    samples: usize,
    seed: u64,
    device: usize,
    output: PathBuf,
    train: TrainOptions,
    verify_only: bool,
    help: bool,
}

impl Options {
    fn parse() -> Result<Self> {
        let mut options = Self {
            depth: 14,
            samples: 1_024,
            seed: 7,
            device: 0,
            output: PathBuf::from("target/mesh-net"),
            train: TrainOptions::default(),
            verify_only: false,
            help: false,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--depth" => options.depth = value(&mut arguments, &argument)?,
                "--samples" => options.samples = value(&mut arguments, &argument)?,
                "--seed" => options.seed = value(&mut arguments, &argument)?,
                "--device" => options.device = value(&mut arguments, &argument)?,
                "--epochs" => options.train.epochs = value(&mut arguments, &argument)?,
                "--learning-rate" => {
                    options.train.learning_rate = value(&mut arguments, &argument)?
                }
                "--report-every" => options.train.report_every = value(&mut arguments, &argument)?,
                "--output" => options.output = value(&mut arguments, &argument)?,
                "--verify-only" => options.verify_only = true,
                "--help" | "-h" => options.help = true,
                _ => return Err(format!("unknown argument: {argument}; use --help").into()),
            }
        }
        Ok(options)
    }
}

fn value<T: FromStr>(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<T> {
    let value = arguments
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))?;
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value}").into())
}

fn main() -> Result<()> {
    let options = Options::parse()?;
    if options.help {
        println!(
            "Mesh Neural Net: Wolfram's ELU diamond, trained with CUDA Oxide\n\
            Usage: cargo oxide run -- [options]\n\n\
            --depth N             Even depth, 2..32 (default: 14)\n\
            --samples N           Training samples, 1..65536 (default: 1024)\n\
            --epochs N            Adam steps (default: 3000)\n\
            --learning-rate X     Positive finite rate (default: 0.01)\n\
            --report-every N      Loss reporting interval (default: 250)\n\
            --seed N              Initialization and data seed (default: 7)\n\
            --device N            CUDA device ordinal (default: 0)\n\
            --output PATH         Artifact directory (default: target/mesh-net)\n\
            --verify-only         Compare GPU forward/backward/Adam with CPU\n\
            --help                Print this help"
        );
        return Ok(());
    }
    if options.verify_only {
        return verify_gpu(options.device);
    }
    let topology = MeshTopology::new(options.depth)?;
    options.train.validate()?;
    if options.samples == 0 || options.samples > 65_536 {
        return Err("samples must be between 1 and 65536".into());
    }
    let mut model = GpuMesh::new(options.device, options.depth, options.seed)?;
    println!(
        "ELU diamond: depth={}, neurons={}, weights={}, biases={}, parameters={}",
        topology.depth(),
        topology.nodes().len(),
        topology.weight_count(),
        topology.nodes().len(),
        topology.parameter_count(),
    );
    let inputs = training_inputs(options.samples, options.seed);
    let targets: Vec<_> = inputs.iter().map(|&x| piecewise_target(x)).collect();
    let grid: Vec<_> = (0..=2_000)
        .map(|i| -3.0 + 6.0 * i as f32 / 2_000.0)
        .collect();
    let grid_targets: Vec<_> = grid.iter().map(|&x| piecewise_target(x)).collect();
    let initial = model.predict(&grid)?;
    let report = model.train(&inputs, &targets, options.train, |point| {
        println!("epoch {:>6}: training MSE = {:.8}", point.epoch, point.mse);
    })?;
    let predictions = model.predict(&grid)?;
    let initial_mse = mse(&initial, &grid_targets);
    let final_mse = mse(&predictions, &grid_targets);
    println!("Evaluation MSE: {initial_mse:.8} -> {final_mse:.8}");
    println!("Training time: {:.3} s", report.elapsed.as_secs_f64());

    std::fs::create_dir_all(&options.output)?;
    let mut loss = writer(options.output.join("loss.csv"))?;
    writeln!(loss, "epoch,mse")?;
    for point in report.history {
        writeln!(loss, "{},{:.10}", point.epoch, point.mse)?;
    }
    loss.flush()?;
    let mut curve = writer(options.output.join("predictions.csv"))?;
    writeln!(curve, "x,target,initial,prediction")?;
    for i in 0..grid.len() {
        writeln!(
            curve,
            "{:.7},{:.1},{:.9},{:.9}",
            grid[i], grid_targets[i], initial[i], predictions[i]
        )?;
    }
    curve.flush()?;
    let parameters = model.parameters()?;
    let mut weights = writer(options.output.join("parameters.csv"))?;
    writeln!(weights, "parameter,value")?;
    for (i, weight) in parameters.iter().enumerate() {
        writeln!(weights, "{i},{weight:.9}")?;
    }
    weights.flush()?;
    let mut nodes = writer(options.output.join("nodes.csv"))?;
    writeln!(
        nodes,
        "node,layer,position,width,left,right,left_weight,right_weight,bias"
    )?;
    for (i, node) in topology.nodes().iter().enumerate() {
        writeln!(
            nodes,
            "{},{},{},{},{},{},{:.9},{:.9},{:.9}",
            i + 1,
            node.layer,
            node.position,
            topology.widths()[node.layer],
            node.parents[0].map_or(-1, |p| p as i64),
            node.parents[1].map_or(-1, |p| p as i64),
            node.weights[0].map_or(0.0, |p| parameters[p]),
            node.weights[1].map_or(0.0, |p| parameters[p]),
            parameters[node.bias],
        )?;
    }
    nodes.flush()?;
    let mut run = writer(options.output.join("run.txt"))?;
    writeln!(
        run,
        "depth={}\nsamples={}\nepochs={}\nseed={}\nlearning_rate={}\ntraining_seconds={:.6}\ninitial_evaluation_mse={initial_mse:.10}\nfinal_evaluation_mse={final_mse:.10}",
        options.depth,
        options.samples,
        options.train.epochs,
        options.seed,
        options.train.learning_rate,
        report.elapsed.as_secs_f64(),
    )?;
    run.flush()?;
    println!("Saved training artifacts to {}", options.output.display());
    Ok(())
}

fn writer(path: PathBuf) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(File::create(path)?))
}

fn training_inputs(samples: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x853c49e6748fea9b);
    (0..samples)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            -3.0 + 6.0 * ((state >> 40) as u32) as f32 / 16_777_216.0
        })
        .collect()
}

fn mse(predictions: &[f32], targets: &[f32]) -> f64 {
    predictions
        .iter()
        .zip(targets)
        .map(|(&prediction, &target)| (f64::from(prediction) - f64::from(target)).powi(2))
        .sum::<f64>()
        / predictions.len() as f64
}

fn check_close(name: &str, expected: f64, actual: f64) -> Result<()> {
    if !expected.is_finite()
        || !actual.is_finite()
        || (expected - actual).abs() > 3e-4 * (1.0 + expected.abs())
    {
        return Err(format!("{name}: expected {expected:.9}, got {actual:.9}").into());
    }
    Ok(())
}

fn verify_gpu(device: usize) -> Result<()> {
    for depth in [2, 6, 14, 32] {
        let reference = ReferenceMesh::new(depth, 7)?;
        let mut model = GpuMesh::new(device, depth, 7)?;
        for samples in [1, 7, 129] {
            let inputs = training_inputs(samples, 42);
            let targets: Vec<_> = inputs.iter().map(|&x| piecewise_target(x)).collect();
            let expected = reference.predict(&inputs)?;
            let actual = model.predict(&inputs)?;
            for (sample, (&cpu, &gpu)) in expected.iter().zip(&actual).enumerate() {
                check_close(
                    &format!("forward depth={depth} sample={sample}"),
                    f64::from(cpu),
                    f64::from(gpu),
                )?;
            }
            let (cpu_loss, cpu_gradient) = reference.loss_and_gradient(&inputs, &targets)?;
            let (gpu_loss, gpu_gradient) = model.loss_and_gradient(&inputs, &targets)?;
            check_close(&format!("loss depth={depth}"), cpu_loss, gpu_loss)?;
            for (parameter, (&cpu, &gpu)) in cpu_gradient.iter().zip(&gpu_gradient).enumerate() {
                check_close(
                    &format!("gradient depth={depth} parameter={parameter}"),
                    cpu,
                    gpu,
                )?;
            }
        }
        if model.predict(&[f32::NAN]).is_ok()
            || model.loss_and_gradient(&[0.0], &[]).is_ok()
            || model.loss_and_gradient(&[], &[]).is_ok()
            || model.loss_and_gradient(&[0.0], &[f32::INFINITY]).is_ok()
            || model.loss_and_gradient(&[0.0], &[f32::MAX]).is_ok()
            || model.set_parameters(&[0.0]).is_ok()
            || model
                .set_parameters(&vec![f32::NAN; reference.parameters().len()])
                .is_ok()
            || !model.predict(&[])?.is_empty()
        {
            return Err("GPU API failed input validation".into());
        }

        let inputs = training_inputs(129, 42);
        let targets: Vec<_> = inputs.iter().map(|&x| piecewise_target(x)).collect();
        let mut parameters = reference.parameters().to_vec();
        let mut first_step = Vec::new();
        let mut moments = vec![[0.0_f32; 2]; parameters.len()];
        let (mut beta1_power, mut beta2_power) = (1.0_f32, 1.0_f32);
        for step in 1..=3 {
            let cpu = ReferenceMesh::from_parameters(depth, parameters.clone())?;
            let (_, gradient) = cpu.loss_and_gradient(&inputs, &targets)?;
            beta1_power *= 0.9;
            beta2_power *= 0.999;
            for ((weight, moment), gradient) in
                parameters.iter_mut().zip(&mut moments).zip(gradient)
            {
                let gradient = gradient as f32;
                moment[0] = 0.9 * moment[0] + 0.1 * gradient;
                moment[1] = 0.999 * moment[1] + 0.001 * gradient * gradient;
                let m = moment[0] / (1.0 - beta1_power);
                let v = moment[1] / (1.0 - beta2_power);
                *weight -= 0.002 * m / (v.sqrt() + 1e-8);
            }
            if step == 1 {
                first_step = parameters.clone();
            }
            model.train(
                &inputs,
                &targets,
                TrainOptions {
                    epochs: 1,
                    learning_rate: 0.002,
                    report_every: 1,
                },
                |_| {},
            )?;
            for (parameter, (&cpu, gpu)) in parameters.iter().zip(model.parameters()?).enumerate() {
                check_close(
                    &format!("Adam depth={depth} step={step} parameter={parameter}"),
                    f64::from(cpu),
                    f64::from(gpu),
                )?;
            }
        }
        model.set_parameters(reference.parameters())?;
        if model.parameters()? != reference.parameters() {
            return Err("parameter round-trip failed".into());
        }
        model.train(
            &inputs,
            &targets,
            TrainOptions {
                epochs: 1,
                learning_rate: 0.002,
                report_every: 1,
            },
            |_| {},
        )?;
        for (parameter, (&cpu, gpu)) in first_step.iter().zip(model.parameters()?).enumerate() {
            check_close(
                &format!("Adam reset depth={depth} parameter={parameter}"),
                f64::from(cpu),
                f64::from(gpu),
            )?;
        }
        println!(
            "PASS depth={depth}: forward, backward, Adam continuation/reset, validation, batch tails"
        );
    }
    println!("CUDA Mesh Neural Net verification passed");
    Ok(())
}
