//! Input-only interpolation control. All 97 output rows remain independently
//! trainable; only K of those rows participate in the operand embeddings.
use grokking_abmod::{
    Result,
    gpt::{AdamWOptions, Equation, EquationDataset, EquationSampler, ModularOperation},
    simple::{
        Architecture, DenseEmbeddingInitialization, GpuSimpleModel, ReferenceSimpleModel,
        SimpleConfig,
    },
};
use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
    time::Instant,
};

const MODULUS: usize = 97;
const METHODS: [&str; 4] = ["linear", "linear-scaled", "cubic", "shepard"];

fn dictionary(k: usize) -> Vec<usize> {
    (0..k).map(|i| (i * 96 + (k - 1) / 2) / (k - 1)).collect()
}

fn codes(anchors: &[usize], method: &str) -> Result<Vec<Vec<f32>>> {
    if anchors.len() < 2
        || anchors.first() != Some(&0)
        || anchors.last() != Some(&96)
        || anchors.windows(2).any(|q| q[0] >= q[1])
        || !METHODS.contains(&method)
    {
        return Err(
            "interpolation requires ordered endpoint-covering anchors and a known method".into(),
        );
    }
    let mut result = Vec::new();
    for value in 0..MODULUS {
        let mut row = vec![0.0_f64; MODULUS];
        if anchors.contains(&value) {
            row[value] = 1.0;
        } else if method == "shepard" {
            let mut nearest = anchors.to_vec();
            nearest.sort_by_key(|&q| (value.abs_diff(q), q));
            for &q in nearest.iter().take(4) {
                row[q] = 1.0 / (value.abs_diff(q) as f64).powi(2);
            }
            let sum: f64 = row.iter().sum();
            for w in &mut row {
                *w /= sum;
            }
        } else {
            let right = anchors.partition_point(|&q| q < value);
            let left = right - 1;
            let (lo, hi) = (anchors[left], anchors[right]);
            let span = (hi - lo) as f64;
            let t = (value - lo) as f64 / span;
            if method == "cubic" {
                // Cubic Hermite interpolation with centered secant tangents.
                // Endpoints use the one available secant; repeated weights add.
                let previous = anchors[left.saturating_sub(1)];
                let next = anchors[(right + 1).min(anchors.len() - 1)];
                let h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
                let h10 = t * t * t - 2.0 * t * t + t;
                let h01 = -2.0 * t * t * t + 3.0 * t * t;
                let h11 = t * t * t - t * t;
                row[lo] += h00;
                row[hi] += h01;
                row[hi] += h10 * span / (hi - previous) as f64;
                row[previous] -= h10 * span / (hi - previous) as f64;
                row[next] += h11 * span / (next - lo) as f64;
                row[lo] -= h11 * span / (next - lo) as f64;
            } else {
                row[lo] = 1.0 - t;
                row[hi] = t;
                if method == "linear-scaled" {
                    let scale = ((1.0 - t).powi(2) + t * t).sqrt();
                    row[lo] /= scale;
                    row[hi] /= scale;
                }
            }
        }
        result.push(row.into_iter().map(|w| w as f32).collect());
    }
    Ok(result)
}

fn encode(equations: &[Equation], codes: &[Vec<f32>]) -> (Vec<f32>, Vec<u32>) {
    let features = equations
        .iter()
        .flat_map(|e| codes[e.a].iter().chain(&codes[e.b]).copied())
        .collect();
    (
        features,
        equations.iter().map(|e| e.answer as u32).collect(),
    )
}

#[derive(Default)]
struct Metrics {
    correct: usize,
    rows: usize,
    loss: f64,
    operands: [[usize; 2]; 3],
    answer: [[usize; 2]; 2],
    all_outside: [usize; 2],
}
impl Metrics {
    fn accuracy(&self) -> f64 {
        self.correct as f64 / self.rows as f64
    }
    fn json(&self) -> String {
        format!(
            "{{\"correct\":{},\"rows\":{},\"accuracy\":{},\"cross_entropy\":{},\"off_dictionary_operands\":{:?},\"off_dictionary_answer\":{:?},\"both_operands_and_answer_outside\":{:?}}}",
            self.correct,
            self.rows,
            self.accuracy(),
            self.loss / self.rows as f64,
            self.operands,
            self.answer,
            self.all_outside
        )
    }
}
fn metrics(
    equations: &[Equation],
    predictions: &[u32],
    logits: &[f32],
    anchors: &[usize],
) -> Result<Metrics> {
    let mut m = Metrics::default();
    for ((e, &p), row) in equations
        .iter()
        .zip(predictions)
        .zip(logits.as_chunks::<97>().0)
    {
        let correct = usize::from(e.answer == p as usize);
        let operands = usize::from(!anchors.contains(&e.a)) + usize::from(!anchors.contains(&e.b));
        let answer = usize::from(!anchors.contains(&e.answer));
        m.rows += 1;
        m.correct += correct;
        m.operands[operands][0] += correct;
        m.operands[operands][1] += 1;
        m.answer[answer][0] += correct;
        m.answer[answer][1] += 1;
        if operands == 2 && answer == 1 {
            m.all_outside[0] += correct;
            m.all_outside[1] += 1;
        }
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m.loss += max
            + row
                .iter()
                .map(|&x| (f64::from(x) - max).exp())
                .sum::<f64>()
                .ln()
            - f64::from(row[e.answer]);
    }
    if m.rows != equations.len() || !m.loss.is_finite() {
        return Err("invalid interpolation evaluation".into());
    }
    Ok(m)
}

fn configuration(width: usize) -> SimpleConfig {
    SimpleConfig {
        architecture: Architecture::TiedRotaryResidualDenseMlp,
        width,
        dense_embedding_initialization: DenseEmbeddingInitialization::NormalWidth,
        ..SimpleConfig::default()
    }
}

fn verify() -> Result<()> {
    let config = configuration(6);
    let dataset = EquationDataset::new(97, ModularOperation::Divide, 0.5, 11)?;
    for method in METHODS {
        let code = codes(&dictionary(7), method)?;
        let (features, targets) = encode(&dataset.split(false)[..3], &code);
        let reference = ReferenceSimpleModel::new(config, 11)?;
        let (wanted_loss, wanted) = reference.loss_and_gradient(&features, &targets)?;
        let mut gpu = GpuSimpleModel::new(0, config, 11)?;
        let table = gpu.upload_table(&features, &targets)?;
        let (loss, gradient) = gpu.loss_and_gradient(&table, &[0, 1, 2])?;
        if (loss - wanted_loss).abs() > 1e-5 {
            return Err(format!("{method} loss mismatch").into());
        }
        for (i, (&a, &b)) in gradient.iter().zip(&wanted).enumerate() {
            if !a.is_finite() || (a - b).abs() > 1e-4 * (1.0 + b.abs()) {
                return Err(format!("{method} gradient {i} mismatch: GPU {a}, f64 {b}").into());
            }
        }
        println!(
            "PASS input interpolation {method}: loss and every parameter gradient vs independent f64 reference"
        );
    }
    Ok(())
}

fn run() -> Result<()> {
    let mut epochs = 2000_usize;
    let mut sizes = vec![97, 49, 25, 13, 7];
    let mut seeds = vec![11_u64, 17, 23];
    let mut methods: Vec<String> = METHODS.iter().map(|s| (*s).to_owned()).collect();
    let mut out = PathBuf::from("target/anchor-interpolation");
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--verify-only" => return verify(),
            "--epochs" => epochs = args.next().ok_or("missing epochs")?.parse()?,
            "--sizes" => {
                sizes = args
                    .next()
                    .ok_or("missing sizes")?
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()?
            }
            "--seeds" => {
                seeds = args
                    .next()
                    .ok_or("missing seeds")?
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()?
            }
            "--methods" => {
                methods = args
                    .next()
                    .ok_or("missing methods")?
                    .split(',')
                    .map(str::to_owned)
                    .collect()
            }
            "--out" => out = args.next().ok_or("missing output directory")?.into(),
            _ => return Err(format!("unknown flag {flag}").into()),
        }
    }
    if epochs == 0
        || sizes.is_empty()
        || seeds.is_empty()
        || methods.is_empty()
        || sizes.iter().any(|&k| !(2..=97).contains(&k))
    {
        return Err("invalid interpolation study settings".into());
    }
    let c = configuration(16);
    let optimizer = AdamWOptions {
        learning_rate: 0.1,
        weight_decay: 0.0,
        beta2: 0.98,
        ..AdamWOptions::default()
    };
    let mut all_summaries = Vec::new();
    for method in &methods {
        let directory = out.join(method);
        fs::create_dir_all(&directory)?;
        let mut summaries = Vec::new();
        for &seed in &seeds {
            let dataset = EquationDataset::new(97, ModularOperation::Divide, 0.5, seed)?;
            for &k in &sizes {
                // Every method is the identical one-hot control at K=97; run it once per seed.
                if k == 97 && method != &methods[0] {
                    continue;
                }
                let anchors = dictionary(k);
                let code = codes(&anchors, method)?;
                let mut model = GpuSimpleModel::new(0, c, seed)?;
                let mut tables = Vec::new();
                for validation in [false, true] {
                    let (features, targets) = encode(dataset.split(validation), &code);
                    tables.push(model.upload_table(&features, &targets)?);
                }
                let mut sampler = EquationSampler::new(tables[0].rows(), seed)?;
                let mut file = File::create(directory.join(format!("k{k}-seed{seed}.csv")))?;
                writeln!(
                    file,
                    "epoch,updates,train_accuracy,test_accuracy,train_cross_entropy,test_cross_entropy,off_dictionary_both_correct,off_dictionary_both_rows,off_dictionary_answer_correct,off_dictionary_answer_rows,all_outside_correct,all_outside_rows,elapsed_seconds"
                )?;
                let started = Instant::now();
                let mut updates = 0;
                let mut first_99 = None;
                let mut best: f64 = 0.0;
                let mut final_metrics = String::new();
                for epoch in 0..=epochs {
                    if epoch > 0 {
                        let mut indices = Vec::with_capacity(tables[0].rows());
                        while indices.len() < tables[0].rows() {
                            indices.extend(sampler.sample(512)?.iter().map(|&i| i as u32));
                        }
                        let batches = model.upload_batches(&tables[0], &indices)?;
                        for offset in (0..indices.len()).step_by(512) {
                            updates += 1;
                            model.train_batch(
                                &tables[0],
                                &batches,
                                offset,
                                512.min(indices.len() - offset),
                                AdamWOptions {
                                    learning_rate: optimizer.learning_rate
                                        * (updates.min(10) as f32 / 10.0),
                                    ..optimizer
                                },
                            )?;
                        }
                    }
                    if !epoch.is_multiple_of(50) && epoch != epochs {
                        continue;
                    }
                    let evaluations = model.evaluate([&tables[0], &tables[1]])?;
                    let train = metrics(
                        dataset.split(false),
                        &evaluations[0].predictions,
                        &evaluations[0].logits,
                        &anchors,
                    )?;
                    let test = metrics(
                        dataset.split(true),
                        &evaluations[1].predictions,
                        &evaluations[1].logits,
                        &anchors,
                    )?;
                    best = best.max(test.accuracy());
                    if test.accuracy() >= 0.99 && first_99.is_none() {
                        first_99 = Some(epoch);
                    }
                    writeln!(
                        file,
                        "{epoch},{updates},{},{},{},{},{},{},{},{},{},{},{}",
                        train.accuracy(),
                        test.accuracy(),
                        train.loss / train.rows as f64,
                        test.loss / test.rows as f64,
                        test.operands[2][0],
                        test.operands[2][1],
                        test.answer[1][0],
                        test.answer[1][1],
                        test.all_outside[0],
                        test.all_outside[1],
                        started.elapsed().as_secs_f64()
                    )?;
                    println!(
                        "method={method} seed={seed} K={k} epoch={epoch} train={:.2}% test={:.2}% elapsed={:.1}s",
                        train.accuracy() * 100.0,
                        test.accuracy() * 100.0,
                        started.elapsed().as_secs_f64()
                    );
                    if epoch == epochs {
                        final_metrics =
                            format!("\"train\":{},\"test\":{}", train.json(), test.json());
                    }
                    for e in evaluations {
                        model.reuse(e.logits);
                    }
                    file.flush()?;
                }
                let p = model.parameters()?;
                let mut checkpoint = File::create(directory.join(format!("k{k}-seed{seed}.f32")))?;
                for value in p {
                    checkpoint.write_all(&value.to_le_bytes())?;
                }
                let summary = format!(
                    "{{\"seed\":{seed},\"dictionary_size\":{k},\"anchors\":{anchors:?},\"method\":\"{method}\",\"rotate\":false,\"input_only\":true,\"parameters\":{},\"epochs\":{epochs},\"updates\":{updates},\"first_recorded_test_99_epoch\":{},\"best_recorded_test_accuracy\":{best},\"elapsed_seconds\":{},{final_metrics}}}",
                    c.parameter_count(),
                    first_99.map_or_else(|| "null".into(), |e| e.to_string()),
                    started.elapsed().as_secs_f64()
                );
                fs::write(directory.join(format!("k{k}-seed{seed}.json")), &summary)?;
                summaries.push(summary.clone());
                all_summaries.push(summary);
                let expected = seeds.len()
                    * (sizes.len() - usize::from(sizes.contains(&97) && method != &methods[0]));
                fs::write(
                    directory.join("summary.json"),
                    format!(
                        "{{\"task\":\"a / b mod 97\",\"status\":\"{}\",\"width\":16,\"learning_rate\":0.1,\"weight_decay\":0,\"input_only\":true,\"method\":\"{method}\",\"output\":\"97 independently trainable class vectors; anchor rows shared with input\",\"membership\":\"fixed uniformly spaced canonical residues\",\"stratum_counts\":\"[correct, rows]\",\"runs\":[{}]}}",
                        if summaries.len() == expected {
                            "complete"
                        } else {
                            "running"
                        },
                        summaries.join(",\n")
                    ),
                )?;
            }
        }
    }
    fs::write(
        out.join("all-runs.json"),
        format!("[{}]\n", all_summaries.join(",\n")),
    )?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interpolation_preserves_anchors_and_distinguishes_all_inputs() {
        for method in METHODS {
            for k in [7, 13, 25, 49, 97] {
                let anchors = dictionary(k);
                let code = codes(&anchors, method).unwrap();
                for (x, row) in code.iter().enumerate() {
                    assert!(row.iter().all(|w| w.is_finite()));
                    if method == "linear-scaled" {
                        assert!((row.iter().map(|w| w * w).sum::<f32>() - 1.0).abs() < 1e-6);
                    } else {
                        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
                    }
                    assert!(
                        row.iter()
                            .enumerate()
                            .all(|(q, &w)| w == 0.0 || anchors.contains(&q))
                    );
                    if anchors.contains(&x) {
                        assert_eq!(row[x], 1.0);
                        assert_eq!(row.iter().filter(|&&w| w != 0.0).count(), 1);
                    }
                    assert!(!code[..x].contains(row));
                }
            }
        }
    }
}
