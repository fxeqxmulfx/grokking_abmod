//! Compare whole-value tables with shared numerical encoders on held-out pairs.
use grokking_abmod::{
    Result,
    gpt::{AdamWOptions, Equation, EquationDataset, EquationSampler, ModularOperation},
    simple::{Architecture, GpuNumericalModel, NumericalCode, SimpleConfig},
};
use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
    time::Instant,
};

#[derive(Default)]
struct Metrics {
    correct: usize,
    rows: usize,
    loss: f64,
    zero_numerator: [usize; 2],
    operand_strata: [[usize; 2]; 4],
}
impl Metrics {
    fn accuracy(&self) -> f64 {
        self.correct as f64 / self.rows as f64
    }
    fn json(&self) -> String {
        format!(
            "{{\"correct\":{},\"rows\":{},\"accuracy\":{},\"cross_entropy\":{},\"zero_numerator\":{:?},\"operand_strata\":{:?}}}",
            self.correct,
            self.rows,
            self.accuracy(),
            self.loss / self.rows as f64,
            self.zero_numerator,
            self.operand_strata
        )
    }
}
fn metrics(
    equations: &[Equation],
    predictions: &[u32],
    logits: &[f32],
    holdout: Option<usize>,
) -> Result<Metrics> {
    if equations.len() != predictions.len() || logits.len() != equations.len() * 97 {
        return Err("invalid evaluation shape".into());
    }
    let mut m = Metrics::default();
    for ((e, &p), row) in equations
        .iter()
        .zip(predictions)
        .zip(logits.as_chunks::<97>().0)
    {
        let correct = usize::from(e.answer == p as usize);
        m.correct += correct;
        m.rows += 1;
        let category = usize::from(holdout.is_some_and(|start| e.a >= start))
            + 2 * usize::from(holdout.is_some_and(|start| e.b >= start));
        m.operand_strata[category][0] += correct;
        m.operand_strata[category][1] += 1;
        if e.a == 0 {
            m.zero_numerator[0] += correct;
            m.zero_numerator[1] += 1;
        }
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m.loss += max
            + row
                .iter()
                .map(|&v| (f64::from(v) - max).exp())
                .sum::<f64>()
                .ln()
            - f64::from(row[e.answer]);
    }
    if !m.loss.is_finite() {
        return Err("nonfinite numerical encoding loss".into());
    }
    Ok(m)
}

struct StudyData {
    train: Vec<Equation>,
    test: Vec<Equation>,
    operand_counts: Vec<usize>,
    answer_counts: Vec<usize>,
}
impl StudyData {
    fn new(operation: ModularOperation, seed: u64, holdout: Option<usize>) -> Result<Self> {
        if holdout.is_some_and(|start| !(2..97).contains(&start)) {
            return Err("held-out operand range must start between 2 and 96".into());
        }
        let original = EquationDataset::new(97, operation, 0.5, seed)?;
        let mut result = Self {
            train: Vec::new(),
            test: original.split(true).to_vec(),
            operand_counts: vec![0; 97],
            answer_counts: vec![0; 97],
        };
        for &e in original.split(false) {
            if holdout.is_some_and(|start| e.a >= start || e.b >= start) {
                result.test.push(e);
            } else {
                result.operand_counts[e.a] += 1;
                result.operand_counts[e.b] += 1;
                result.answer_counts[e.answer] += 1;
                result.train.push(e);
            }
        }
        if result.train.len() < 2 || holdout.is_some() && result.answer_counts.contains(&0) {
            return Err(
                "operand holdout needs training equations and exposure of every answer class"
                    .into(),
            );
        }
        Ok(result)
    }
}

fn parse_list<T: std::str::FromStr>(value: Option<String>) -> Result<Vec<T>>
where
    T::Err: std::error::Error + 'static,
{
    value
        .ok_or("missing list")?
        .split(',')
        .map(|v| v.parse().map_err(Into::into))
        .collect()
}
fn run() -> Result<()> {
    let mut epochs = 2000_usize;
    let mut widths = vec![16_usize, 64];
    let mut seeds = vec![11_u64, 17, 23];
    let mut methods: Vec<String> = NumericalCode::NAMES.iter().map(|s| (*s).into()).collect();
    let mut encoders: Vec<String> = vec!["linear".into(), "mlp".into()];
    let mut harmonics = 8_usize;
    let mut learning_rate = 0.1_f32;
    let mut weight_decay = 0.0_f32;
    let mut operation = ModularOperation::Divide;
    let mut split = String::from("pairs");
    let mut holdout_start = 80_usize;
    let mut out = PathBuf::from("target/numerical-div97");
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--verify-only" => return GpuNumericalModel::verify(0),
            "--epochs" => epochs = args.next().ok_or("missing epochs")?.parse()?,
            "--widths" => widths = parse_list(args.next())?,
            "--seeds" => seeds = parse_list(args.next())?,
            "--methods" => {
                methods = args
                    .next()
                    .ok_or("missing methods")?
                    .split(',')
                    .map(str::to_owned)
                    .collect()
            }
            "--encoders" => {
                encoders = args
                    .next()
                    .ok_or("missing encoders")?
                    .split(',')
                    .map(str::to_owned)
                    .collect()
            }
            "--harmonics" => harmonics = args.next().ok_or("missing harmonics")?.parse()?,
            "--lr" => learning_rate = args.next().ok_or("missing learning rate")?.parse()?,
            "--wd" => weight_decay = args.next().ok_or("missing weight decay")?.parse()?,
            "--operation" => operation = args.next().ok_or("missing operation")?.parse()?,
            "--split" => split = args.next().ok_or("missing split")?,
            "--holdout-start" => {
                holdout_start = args.next().ok_or("missing holdout start")?.parse()?
            }
            "--out" => out = args.next().ok_or("missing output")?.into(),
            _ => return Err(format!("unknown flag {flag}").into()),
        }
    }
    if epochs == 0
        || widths.is_empty()
        || seeds.is_empty()
        || methods.is_empty()
        || encoders.is_empty()
        || encoders.iter().any(|s| s != "linear" && s != "mlp")
        || !["pairs", "operands"].contains(&split.as_str())
    {
        return Err("invalid numerical encoding study".into());
    }
    let optimizer = AdamWOptions {
        learning_rate,
        weight_decay,
        beta2: 0.98,
        ..AdamWOptions::default()
    };
    optimizer.validate()?;
    let holdout = (split == "operands").then_some(holdout_start);
    let holdout_json = holdout.map_or_else(|| "null".into(), |start| start.to_string());
    fs::create_dir_all(&out)?;
    let expected_runs = widths.len() * seeds.len() * methods.len() * encoders.len();
    let mut summaries = Vec::new();
    for &width in &widths {
        let config = SimpleConfig {
            architecture: Architecture::ResidualMlp,
            width,
            expansion: 1,
            depth: 1,
            ..SimpleConfig::default()
        };
        for method in &methods {
            let code = NumericalCode::new(97, method, harmonics)?;
            if method == "random-bits" || method == "integer-bits" {
                let mut file = File::create(out.join(format!("{method}-h{harmonics}-codes.csv")))?;
                writeln!(file, "value,bits")?;
                for (value, row) in code.values().chunks_exact(code.features()).enumerate() {
                    let bits: String = row
                        .iter()
                        .take(code.features() - usize::from(method == "integer-bits"))
                        .map(|&v| if v > 0.0 { '1' } else { '0' })
                        .collect();
                    writeln!(file, "{value},{bits}")?;
                }
            }
            for encoder in &encoders {
                for &seed in &seeds {
                    let suffix =
                        holdout.map_or_else(String::new, |start| format!("-holdout{start}"));
                    let name =
                        format!("{method}-{encoder}-w{width}-h{harmonics}-seed{seed}{suffix}");
                    let dataset = StudyData::new(operation, seed, holdout)?;
                    let mut model =
                        GpuNumericalModel::new(0, config, code.clone(), encoder == "mlp", seed)?;
                    let mut tables = Vec::new();
                    for equations in [&dataset.train, &dataset.test] {
                        let (features, targets) = config.encode(equations)?;
                        tables.push(model.upload_table(&features, &targets)?);
                    }
                    let mut sampler = EquationSampler::new(tables[0].rows(), seed)?;
                    let mut file = File::create(out.join(format!("{name}.csv")))?;
                    writeln!(
                        file,
                        "epoch,updates,train_accuracy,test_accuracy,train_cross_entropy,test_cross_entropy,zero_numerator_correct,zero_numerator_rows,familiar_correct,familiar_rows,unseen_a_correct,unseen_a_rows,unseen_b_correct,unseen_b_rows,both_unseen_correct,both_unseen_rows,elapsed_seconds"
                    )?;
                    let started = Instant::now();
                    let mut updates = 0;
                    let mut best = 0.0_f64;
                    let mut first_99 = None;
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
                            &dataset.train,
                            &evaluations[0].predictions,
                            &evaluations[0].logits,
                            holdout,
                        )?;
                        let test = metrics(
                            &dataset.test,
                            &evaluations[1].predictions,
                            &evaluations[1].logits,
                            holdout,
                        )?;
                        best = best.max(test.accuracy());
                        if test.accuracy() >= 0.99 && first_99.is_none() {
                            first_99 = Some(epoch);
                        }
                        writeln!(
                            file,
                            "{epoch},{updates},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                            train.accuracy(),
                            test.accuracy(),
                            train.loss / train.rows as f64,
                            test.loss / test.rows as f64,
                            test.zero_numerator[0],
                            test.zero_numerator[1],
                            test.operand_strata[0][0],
                            test.operand_strata[0][1],
                            test.operand_strata[1][0],
                            test.operand_strata[1][1],
                            test.operand_strata[2][0],
                            test.operand_strata[2][1],
                            test.operand_strata[3][0],
                            test.operand_strata[3][1],
                            started.elapsed().as_secs_f64()
                        )?;
                        let extra = if holdout.is_some() {
                            let [correct, rows] = test.operand_strata[3];
                            format!(" both_unseen={:.2}%", 100.0 * correct as f64 / rows as f64)
                        } else {
                            String::new()
                        };
                        println!(
                            "{name} epoch={epoch} train={:.2}% test={:.2}%{extra} elapsed={:.1}s",
                            100.0 * train.accuracy(),
                            100.0 * test.accuracy(),
                            started.elapsed().as_secs_f64()
                        );
                        if epoch == epochs {
                            final_metrics =
                                format!("\"train\":{},\"test\":{}", train.json(), test.json());
                        }
                        for evaluation in evaluations {
                            model.reuse(evaluation);
                        }
                        file.flush()?;
                    }
                    let p = model.parameters()?;
                    let mut checkpoint = File::create(out.join(format!("{name}.f32")))?;
                    for value in p {
                        checkpoint.write_all(&value.to_le_bytes())?;
                    }
                    let summary = format!(
                        "{{\"name\":\"{name}\",\"method\":\"{method}\",\"encoder\":\"{encoder}\",\"seed\":{seed},\"width\":{width},\"harmonics\":{harmonics},\"split\":\"{split}\",\"heldout_start\":{holdout_json},\"training_operand_counts\":{:?},\"training_answer_counts\":{:?},\"code_features\":{},\"whole_value_input_rows\":{},\"parameters\":{},\"epochs\":{epochs},\"updates\":{updates},\"first_recorded_test_99_epoch\":{},\"best_recorded_test_accuracy\":{best},\"elapsed_seconds\":{},{final_metrics}}}",
                        dataset.operand_counts,
                        dataset.answer_counts,
                        code.features(),
                        if method == "table" { 97 } else { 0 },
                        model.parameter_count(),
                        first_99.map_or_else(|| "null".into(), |e| e.to_string()),
                        started.elapsed().as_secs_f64()
                    );
                    fs::write(out.join(format!("{name}.json")), &summary)?;
                    summaries.push(summary);
                    fs::write(
                        out.join("summary.json"),
                        format!(
                            "{{\"status\":\"{}\",\"operation\":\"{operation}\",\"modulus\":97,\"split\":\"{split}\",\"heldout_start\":{holdout_json},\"original_train_fraction\":0.5,\"learning_rate\":{learning_rate},\"weight_decay\":{weight_decay},\"beta1\":0.9,\"beta2\":0.98,\"warmup_updates\":10,\"batch_size\":512,\"core\":\"one pre-RMSNorm residual FFN, expansion 1, ReLU, final RMSNorm\",\"operand_combination\":\"shared e(a) + Rotate90(e(b))\",\"output\":\"97 independent trainable class vectors\",\"code_norm\":\"unit except integer-bits: [MSB-first bits, 1] scaled by 1/sqrt(bit_count+1)\",\"random_code_seed\":\"fixed independently of split and model seeds\",\"split_scope\":\"pairs: held-out equations; operands: upper values excluded from both training operands but still exposed as answer classes\",\"operand_strata_order\":[\"familiar\",\"unseen_a\",\"unseen_b\",\"both_unseen\"],\"stratum_counts\":\"[correct, rows]\",\"runs\":[{}]}}",
                            if summaries.len() == expected_runs {
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
    }
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
    use std::collections::BTreeSet;

    #[test]
    fn pair_control_preserves_the_original_seeded_split() {
        let original = EquationDataset::new(97, ModularOperation::Divide, 0.5, 11).unwrap();
        let study = StudyData::new(ModularOperation::Divide, 11, None).unwrap();
        assert_eq!(study.train, original.split(false));
        assert_eq!(study.test, original.split(true));
    }

    #[test]
    fn operand_holdout_is_complete_disjoint_and_keeps_all_answer_classes() {
        for seed in [11, 17, 23] {
            let data = StudyData::new(ModularOperation::Divide, seed, Some(80)).unwrap();
            assert!(data.train.iter().all(|e| e.a < 80 && e.b < 80));
            assert!(data.operand_counts[80..].iter().all(|&n| n == 0));
            assert!(data.operand_counts[..80].iter().all(|&n| n > 0));
            assert!(data.answer_counts.iter().all(|&n| n > 0));
            assert_eq!(data.train.len() + data.test.len(), 97 * 96);
            let pairs: BTreeSet<_> = data
                .train
                .iter()
                .chain(&data.test)
                .map(|e| (e.a, e.b))
                .collect();
            assert_eq!(pairs.len(), 97 * 96);
            assert_eq!(
                data.test.iter().filter(|e| e.a >= 80 && e.b >= 80).count(),
                17 * 17
            );
        }
    }
}
