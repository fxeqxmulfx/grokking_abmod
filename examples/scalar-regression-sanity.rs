//! End-to-end CUDA control: learn a known affine function from numeric pairs.
use grokking_abmod::{
    Result,
    gpt::{AdamWOptions, EquationSampler},
    simple::{GpuRegressionMlp, RegressionConfig},
};

fn table(start: usize) -> (Vec<f32>, Vec<f32>) {
    let features: Vec<_> = (start..start + 37)
        .flat_map(|row| [(row * 7 % 97) as f32 / 96.0, (row * 11 % 89) as f32 / 88.0])
        .collect();
    let targets = features
        .as_chunks::<2>()
        .0
        .iter()
        .map(|row| 0.2 + 0.3 * row[0] + 0.4 * row[1])
        .collect();
    (features, targets)
}

fn mse(actual: &[f32], expected: &[f32]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (f64::from(actual) - f64::from(expected)).powi(2))
        .sum::<f64>()
        / actual.len() as f64
}

fn run() -> Result<()> {
    let config = RegressionConfig {
        width: 16,
        depth: 1,
        pre_rmsnorm: false,
        residual: false,
    };
    let mut model = GpuRegressionMlp::new(0, config, 11)?;
    let (features, targets) = table(0);
    let train = model.upload_table(&features, &targets)?;
    let (test_features, test_targets) = table(37);
    let test = model.upload_table(&test_features, &test_targets)?;
    let initial = mse(&model.predictions(&test)?, &test_targets);
    let mut sampler = EquationSampler::new(37, 11)?;
    let mut indices: Vec<_> = (0..37).collect();
    let mut batches = model.upload_batches(&train, &indices)?;
    let mut updates = 0;
    for _ in 0..1000 {
        indices.clear();
        while indices.len() < 37 {
            indices.extend(sampler.sample(16)?.iter().map(|&index| index as u32));
        }
        model.update_batches(&train, &mut batches, &indices)?;
        for offset in (0..37).step_by(16) {
            updates += 1;
            model.train_batch(
                &train,
                &batches,
                offset,
                16.min(37 - offset),
                AdamWOptions {
                    learning_rate: 0.003 * (updates.min(10) as f32 / 10.0),
                    weight_decay: 0.0,
                    beta2: 0.98,
                    ..AdamWOptions::default()
                },
            )?;
        }
    }
    let final_loss = mse(&model.predictions(&test)?, &test_targets);
    if final_loss >= 1e-4 || final_loss >= initial / 100.0 {
        return Err(format!(
            "affine CUDA control failed to learn: initial {initial}, final {final_loss}"
        )
        .into());
    }
    println!(
        "{{\"function\":\"0.2 + 0.3*a + 0.4*b\",\"width\":16,\"depth\":1,\"epochs\":1000,\"updates\":{updates},\"initialHeldoutMse\":{initial},\"finalHeldoutMse\":{final_loss},\"passed\":true}}"
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
