//! Controlled vocabulary reduction in the established CUDA div97 Dense model.
use grokking_abmod::{
    Result,
    gpt::{AdamWOptions, Equation, EquationDataset, EquationSampler, ModularOperation},
    simple::{
        AnchorEncoding, Architecture, DenseEmbeddingInitialization, GpuAnchoredModel, SimpleConfig,
    },
};
use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
    time::Instant,
};

struct Options {
    epochs: usize,
    report: usize,
    width: usize,
    seeds: Vec<u64>,
    sizes: Vec<usize>,
    learning_rate: f32,
    weight_decay: f32,
    rotate: bool,
    verify: bool,
    out: PathBuf,
}

impl Options {
    fn parse() -> Result<Self> {
        let mut options = Self {
            epochs: 2000,
            report: 50,
            width: 16,
            seeds: vec![11, 17, 23],
            sizes: vec![97, 49, 25, 13, 7],
            learning_rate: 0.1,
            weight_decay: 0.0,
            rotate: true,
            verify: false,
            out: "target/anchor-div97".into(),
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(flag) = arguments.next() {
            match flag.as_str() {
                "--verify-only" => options.verify = true,
                "--no-rotation" => options.rotate = false,
                "--epochs" => options.epochs = arguments.next().ok_or("missing epochs")?.parse()?,
                "--report-every" => {
                    options.report = arguments.next().ok_or("missing report interval")?.parse()?
                }
                "--width" => options.width = arguments.next().ok_or("missing width")?.parse()?,
                "--lr" => {
                    options.learning_rate =
                        arguments.next().ok_or("missing learning rate")?.parse()?
                }
                "--wd" => {
                    options.weight_decay =
                        arguments.next().ok_or("missing weight decay")?.parse()?
                }
                "--seeds" => {
                    options.seeds = arguments
                        .next()
                        .ok_or("missing seeds")?
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?
                }
                "--sizes" => {
                    options.sizes = arguments
                        .next()
                        .ok_or("missing sizes")?
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?
                }
                "--out" => options.out = arguments.next().ok_or("missing output directory")?.into(),
                _ => return Err(format!("unknown option {flag}").into()),
            }
        }
        if options.epochs == 0
            || options.report == 0
            || options.seeds.is_empty()
            || options.sizes.is_empty()
            || options.sizes.iter().any(|&k| k == 0 || k > 97)
        {
            return Err(
                "epochs/report must be positive and dictionary sizes must be 1..=97".into(),
            );
        }
        Ok(options)
    }
}

fn anchors(k: usize) -> Vec<usize> {
    if k == 1 {
        vec![0]
    } else {
        (0..k).map(|i| (i * 96 + (k - 1) / 2) / (k - 1)).collect()
    }
}

#[derive(Default)]
struct Metrics {
    rows: usize,
    correct: usize,
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
            "{{\"rows\":{},\"correct\":{},\"accuracy\":{},\"cross_entropy\":{},\"off_dictionary_operands\":{:?},\"off_dictionary_answer\":{:?},\"both_operands_and_answer_outside\":{:?}}}",
            self.rows,
            self.correct,
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
    dictionary: &[usize],
) -> Result<Metrics> {
    let mut result = Metrics::default();
    for ((e, &prediction), row) in equations
        .iter()
        .zip(predictions)
        .zip(logits.as_chunks::<97>().0)
    {
        let correct = usize::from(prediction as usize == e.answer);
        let outside =
            usize::from(!dictionary.contains(&e.a)) + usize::from(!dictionary.contains(&e.b));
        let answer = usize::from(!dictionary.contains(&e.answer));
        result.rows += 1;
        result.correct += correct;
        result.operands[outside][0] += correct;
        result.operands[outside][1] += 1;
        result.answer[answer][0] += correct;
        result.answer[answer][1] += 1;
        if outside == 2 && answer == 1 {
            result.all_outside[0] += correct;
            result.all_outside[1] += 1;
        }
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = row.iter().map(|&x| (f64::from(x) - maximum).exp()).sum();
        result.loss += maximum + sum.ln() - f64::from(row[e.answer]);
    }
    if result.rows != equations.len() || !result.loss.is_finite() {
        return Err("incomplete or nonfinite anchor evaluation".into());
    }
    Ok(result)
}

fn run(options: Options) -> Result<()> {
    if options.verify {
        return GpuAnchoredModel::verify(0);
    }
    fs::create_dir_all(&options.out)?;
    let config = SimpleConfig {
        architecture: Architecture::TiedRotaryResidualDenseMlp,
        width: options.width,
        depth: 1,
        dense_embedding_initialization: DenseEmbeddingInitialization::NormalWidth,
        ..SimpleConfig::default()
    };
    config.validate()?;
    let optimizer = AdamWOptions {
        learning_rate: options.learning_rate,
        weight_decay: options.weight_decay,
        beta2: 0.98,
        ..AdamWOptions::default()
    };
    optimizer.validate()?;
    let mut summaries = Vec::new();
    for &seed in &options.seeds {
        let dataset = EquationDataset::new(97, ModularOperation::Divide, 0.5, seed)?;
        for &k in &options.sizes {
            let dictionary = anchors(k);
            let encoding =
                AnchorEncoding::new(97, options.width, dictionary.clone(), options.rotate)?;
            let mut model = GpuAnchoredModel::new(0, config, encoding, seed)?;
            let mut tables = Vec::new();
            for validation in [false, true] {
                let (features, targets) = config.encode(dataset.split(validation))?;
                tables.push(model.upload_table(&features, &targets)?);
            }
            let mut sampler = EquationSampler::new(tables[0].rows(), seed)?;
            let mut history = File::create(options.out.join(format!("k{k}-seed{seed}.csv")))?;
            writeln!(
                history,
                "epoch,updates,train_accuracy,test_accuracy,train_cross_entropy,test_cross_entropy,off_dictionary_both_correct,off_dictionary_both_rows,off_dictionary_answer_correct,off_dictionary_answer_rows,all_outside_correct,all_outside_rows,elapsed_seconds"
            )?;
            let started = Instant::now();
            let mut updates = 0;
            let mut first_99 = None;
            let mut final_metrics = String::new();
            let mut best_accuracy: f64 = 0.0;
            for epoch in 0..=options.epochs {
                if epoch > 0 {
                    let mut indices = Vec::with_capacity(tables[0].rows());
                    while indices.len() < tables[0].rows() {
                        indices.extend(sampler.sample(512)?.iter().map(|&i| i as u32));
                    }
                    let batches = model.upload_batches(&tables[0], &indices)?;
                    for offset in (0..indices.len()).step_by(512) {
                        updates += 1;
                        let warmup = (updates.min(10) as f32) / 10.0;
                        model.train_batch(
                            &tables[0],
                            &batches,
                            offset,
                            512.min(indices.len() - offset),
                            AdamWOptions {
                                learning_rate: options.learning_rate * warmup,
                                ..optimizer
                            },
                        )?;
                    }
                }
                if epoch % options.report != 0 && epoch != options.epochs {
                    continue;
                }
                let evaluations = model.evaluate([&tables[0], &tables[1]])?;
                let train = metrics(
                    dataset.split(false),
                    &evaluations[0].predictions,
                    &evaluations[0].logits,
                    &dictionary,
                )?;
                let test = metrics(
                    dataset.split(true),
                    &evaluations[1].predictions,
                    &evaluations[1].logits,
                    &dictionary,
                )?;
                best_accuracy = best_accuracy.max(test.accuracy());
                if test.accuracy() >= 0.99 && first_99.is_none() {
                    first_99 = Some(epoch);
                }
                writeln!(
                    history,
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
                    "seed={seed} K={k} rotate={} epoch={epoch} updates={updates} train={:.2}% test={:.2}% elapsed={:.1}s",
                    options.rotate,
                    100.0 * train.accuracy(),
                    100.0 * test.accuracy(),
                    started.elapsed().as_secs_f64()
                );
                if epoch == options.epochs {
                    final_metrics = format!("\"train\":{},\"test\":{}", train.json(), test.json());
                }
                for evaluation in evaluations {
                    model.reuse(evaluation);
                }
                history.flush()?;
            }
            let parameters = model.parameters()?;
            let mut checkpoint = File::create(options.out.join(format!("k{k}-seed{seed}.f32")))?;
            for p in parameters {
                checkpoint.write_all(&p.to_le_bytes())?;
            }
            let summary = format!(
                "{{\"seed\":{seed},\"dictionary_size\":{k},\"anchors\":{dictionary:?},\"rotate\":{},\"parameters\":{},\"epochs\":{},\"updates\":{updates},\"first_recorded_test_99_epoch\":{},\"best_recorded_test_accuracy\":{best_accuracy},\"elapsed_seconds\":{},{final_metrics}}}",
                options.rotate,
                model.parameter_count(),
                options.epochs,
                first_99.map_or_else(|| "null".into(), |e| e.to_string()),
                started.elapsed().as_secs_f64()
            );
            fs::write(options.out.join(format!("k{k}-seed{seed}.json")), &summary)?;
            summaries.push(summary);
            let all = format!(
                "{{\"task\":\"a / b mod 97\",\"status\":\"{}\",\"width\":{},\"learning_rate\":{},\"weight_decay\":{},\"rope_base\":10000,\"routing\":\"ordinary integer distance; ties choose lower anchor\",\"membership\":\"fixed uniformly spaced anchors\",\"output\":\"97 categorical logits from the same generated embedding table\",\"split\":\"same seeded 50/50 equation split as established div97 baseline\",\"exposure\":\"all 97 identities appear in training; off-dictionary is not unseen in training\",\"runtime_cache\":\"97 derived rows retained on GPU; only K rows independently trainable\",\"stratum_counts\":\"[correct, rows]\",\"runs\":[{}]}}",
                if summaries.len() == options.seeds.len() * options.sizes.len() {
                    "complete"
                } else {
                    "running"
                },
                options.width,
                options.learning_rate,
                options.weight_decay,
                summaries.join(",\n")
            );
            fs::write(options.out.join("summary.json"), all)?;
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = Options::parse().and_then(run) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
