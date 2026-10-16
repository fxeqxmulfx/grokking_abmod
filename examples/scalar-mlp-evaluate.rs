//! Reload a saved scalar MLP and export independent train/held-out predictions.
use grokking_abmod::{
    Result,
    gpt::{Equation, EquationDataset, ModularOperation},
    simple::{GpuRegressionMlp, RegressionConfig},
};
use std::{
    collections::BTreeMap,
    fs::{File, read, read_to_string},
    io::{BufWriter, Write},
    path::Path,
    str::FromStr,
};

fn field<T: FromStr>(manifest: &BTreeMap<String, String>, name: &str) -> Result<T> {
    manifest
        .get(name)
        .ok_or_else(|| format!("missing manifest field {name}"))?
        .parse()
        .map_err(|_| format!("invalid manifest field {name}").into())
}

struct Metrics {
    rows: usize,
    mse: f64,
    rmse: f64,
    accuracy: f64,
}

fn evaluate(
    model: &mut GpuRegressionMlp,
    equations: &[Equation],
    divisor: f32,
    offset: f32,
    output: &Path,
) -> Result<Metrics> {
    let features: Vec<_> = equations
        .iter()
        .flat_map(|row| {
            [
                row.a as f32 / divisor - offset,
                row.b as f32 / divisor - offset,
            ]
        })
        .collect();
    let targets: Vec<_> = equations
        .iter()
        .map(|row| row.answer as f32 / divisor - offset)
        .collect();
    let table = model.upload_table(&features, &targets)?;
    let predictions = model.predictions(&table)?;
    if predictions.len() != equations.len() || predictions.iter().any(|value| !value.is_finite()) {
        return Err("checkpoint produced nonfinite or incomplete predictions".into());
    }
    let mut csv = BufWriter::new(File::create(output)?);
    writeln!(csv, "a,b,target,prediction_residue")?;
    let (mut squares, mut mse, mut correct) = (0.0, 0.0, 0usize);
    for ((row, &prediction), &target) in equations.iter().zip(&predictions).zip(&targets) {
        let restored = (f64::from(prediction) + f64::from(offset)) * f64::from(divisor);
        let error = restored - row.answer as f64;
        squares += error * error;
        mse += (f64::from(prediction) - f64::from(target)).powi(2);
        correct += usize::from(restored.round() == row.answer as f64);
        writeln!(csv, "{},{},{},{restored:.17}", row.a, row.b, row.answer)?;
    }
    csv.flush()?;
    let rows = equations.len();
    Ok(Metrics {
        rows,
        mse: mse / rows as f64,
        rmse: (squares / rows as f64).sqrt(),
        accuracy: correct as f64 / rows as f64,
    })
}

fn run() -> Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&arguments.len()) {
        return Err(
            "usage: scalar-mlp-evaluate RUN_DIRECTORY NEW_OUTPUT_DIRECTORY [DEVICE]".into(),
        );
    }
    let directory = Path::new(&arguments[0]);
    let output = Path::new(&arguments[1]);
    if output.exists() {
        return Err("evaluation output already exists; preserve earlier evidence".into());
    }
    let device = arguments
        .get(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(0);
    let mut manifest = BTreeMap::new();
    for line in read_to_string(directory.join("run.txt"))?.lines() {
        let (key, value) = line.split_once('=').ok_or("invalid manifest line")?;
        manifest.insert(key.to_owned(), value.to_owned());
    }
    for (key, value) in [
        ("architecture", "scalar-mlp"),
        ("status", "complete"),
        ("inputs", "2"),
        ("outputs", "1"),
        ("activation", "relu"),
        ("loss", "mse_mean"),
        ("final_rmsnorm", "false"),
        ("normalization_affine", "false"),
    ] {
        if manifest.get(key).map(String::as_str) != Some(value) {
            return Err(format!("unsupported checkpoint field {key}").into());
        }
    }
    let config = RegressionConfig {
        width: field(&manifest, "width")?,
        depth: field(&manifest, "depth")?,
        pre_rmsnorm: field(&manifest, "pre_rmsnorm")?,
        residual: field(&manifest, "residual")?,
    };
    config.validate()?;
    let bytes = read(directory.join("model.bin"))?;
    if bytes.len() != config.parameter_count() * 4 {
        return Err("checkpoint byte length differs from its architecture".into());
    }
    let parameters: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    let mut model = GpuRegressionMlp::from_parameters(device, config, &parameters)?;
    let modulus = field(&manifest, "modulus")?;
    let operation: ModularOperation = field(&manifest, "operation")?;
    if !matches!(
        operation,
        ModularOperation::Divide | ModularOperation::Multiply
    ) {
        return Err("evaluation supports the division and multiplication experiments".into());
    }
    let seed = field(&manifest, "seed")?;
    let dataset = EquationDataset::new(
        modulus,
        operation,
        field(&manifest, "train_fraction")?,
        seed,
    )?;
    let training_rows = field(&manifest, "training_equations")?;
    if !(1..=dataset.split(false).len()).contains(&training_rows)
        || dataset.split(true).len() != field::<usize>(&manifest, "validation_equations")?
    {
        return Err("checkpoint dataset counts are inconsistent".into());
    }
    let (divisor, offset) = match manifest.get("scale").map(String::as_str) {
        Some("raw") => (1.0, 0.0),
        Some("unit") => ((modulus - 1) as f32, 0.0),
        Some("centered") => ((modulus - 1) as f32, 0.5),
        _ => return Err("unsupported numeric scale".into()),
    };
    std::fs::create_dir_all(output)?;
    let train = evaluate(
        &mut model,
        &dataset.split(false)[..training_rows],
        divisor,
        offset,
        &output.join("train-predictions.csv"),
    )?;
    let test = evaluate(
        &mut model,
        dataset.split(true),
        divisor,
        offset,
        &output.join("test-predictions.csv"),
    )?;
    println!(
        "{{\"architecture\":\"scalar-mlp\",\"operation\":\"{operation}\",\"modulus\":{modulus},\"seed\":{seed},\"width\":{},\"depth\":{},\"parameters\":{},\"unusedTrainingRows\":{},\"train\":{{\"rows\":{},\"mse\":{},\"rmse\":{},\"accuracy\":{}}},\"test\":{{\"rows\":{},\"mse\":{},\"rmse\":{},\"accuracy\":{}}}}}",
        config.width,
        config.depth,
        config.parameter_count(),
        dataset.split(false).len() - training_rows,
        train.rows,
        train.mse,
        train.rmse,
        train.accuracy,
        test.rows,
        test.mse,
        test.rmse,
        test.accuracy
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
