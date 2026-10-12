//! Host-only scheduler for CUDA mesh experiments. Every trial starts from scratch.

use std::collections::HashSet;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Debug)]
struct Policy {
    epochs: u64,
    plateau_epochs: f64,
    jump_epochs: f64,
    train_accuracy: f64,
    low_accuracy: f64,
    high_accuracy: f64,
    final_accuracy: f64,
    search_seed: u64,
    timeout_seconds: u64,
    report_every: u64,
    queue_sha256: Option<String>,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            epochs: 10_000,
            plateau_epochs: 1_000.0,
            jump_epochs: 250.0,
            train_accuracy: 0.99,
            low_accuracy: 0.10,
            high_accuracy: 0.90,
            final_accuracy: 0.99,
            search_seed: 20_261_012,
            timeout_seconds: 3_600,
            report_every: 250,
            queue_sha256: None,
        }
    }
}

impl Policy {
    fn json(&self) -> String {
        format!(
            "{{\"epochs\":{},\"plateau_epochs\":{},\"jump_epochs\":{},\"train_accuracy\":{},\"low_accuracy\":{},\"high_accuracy\":{},\"final_accuracy\":{},\"search_seed\":{},\"timeout_seconds\":{},\"batch\":512,\"report_every_updates\":{},\"report_every_epochs\":{},\"queue_sha256\":{},\"operation\":\"div\",\"modulus\":97,\"train_fraction\":0.5,\"scheduler_version\":2}}",
            self.epochs,
            self.plateau_epochs,
            self.jump_epochs,
            self.train_accuracy,
            self.low_accuracy,
            self.high_accuracy,
            self.final_accuracy,
            self.search_seed,
            self.timeout_seconds,
            self.report_every,
            self.report_every / 10,
            self.queue_sha256
                .as_deref()
                .map_or_else(|| "null".into(), json_string),
        )
    }
}

struct Options {
    output: PathBuf,
    trainer: PathBuf,
    policy: Policy,
    dry_run: u64,
    max_trials: u64,
    stop_after_matches: u64,
    queue: Option<PathBuf>,
}

impl Options {
    fn parse() -> Result<Option<Self>> {
        let mut options = Self {
            output: "target/grok-div97-search/sharp-mesh-loop".into(),
            trainer: "target/release/mesh-gpt".into(),
            policy: Policy::default(),
            dry_run: 0,
            max_trials: 0,
            stop_after_matches: 0,
            queue: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            if flag == "--help" || flag == "-h" {
                println!(
                    "Usage: sharp-mesh-search [OPTIONS]\n\
                     --output PATH              Journal and trial directories\n\
                     --trainer PATH             CUDA mesh-gpt binary to freeze\n\
                     --epochs N                 Per-trial epoch cap (10000)\n\
                     --plateau-epochs N         Minimum contiguous plateau (1000)\n\
                     --jump-epochs N            Maximum sampled 10% -> 90% rise (250)\n\
                     --search-seed N            Reproducible proposal seed (20261012)\n\
                     --report-every N           Full-split evaluation interval in updates (250); multiple of 10\n\
                     --queue PATH               Prioritized mesh argument lines before random proposals\n\
                     --trial-timeout-seconds N  Wall-time cap, including evaluation (3600)\n\
                     --max-trials N             New trials per invocation; 0 = unlimited\n\
                     --stop-after-matches N     Total matches before stopping; 0 = unlimited\n\
                     --dry-run N                Print N initial proposals without starting CUDA\n\
                     Create OUTPUT/STOP to stop the current trial and the loop.\n\
                     Restarting skips finished trials and retries interrupted ones from scratch."
                );
                return Ok(None);
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--output" => options.output = value.into(),
                "--trainer" => options.trainer = value.into(),
                "--epochs" => options.policy.epochs = value.parse()?,
                "--plateau-epochs" => options.policy.plateau_epochs = value.parse()?,
                "--jump-epochs" => options.policy.jump_epochs = value.parse()?,
                "--search-seed" => options.policy.search_seed = value.parse()?,
                "--report-every" => options.policy.report_every = value.parse()?,
                "--queue" => options.queue = Some(value.into()),
                "--trial-timeout-seconds" => options.policy.timeout_seconds = value.parse()?,
                "--max-trials" => options.max_trials = value.parse()?,
                "--stop-after-matches" => options.stop_after_matches = value.parse()?,
                "--dry-run" => options.dry_run = value.parse()?,
                _ => return Err(format!("unknown option: {flag}").into()),
            }
        }
        let p = &options.policy;
        if p.epochs == 0
            || p.timeout_seconds == 0
            || !p.plateau_epochs.is_finite()
            || p.plateau_epochs <= 0.0
            || !p.jump_epochs.is_finite()
            || p.jump_epochs <= 0.0
            || p.report_every == 0
            || !p.report_every.is_multiple_of(10)
        {
            return Err(
                "limits must be positive and finite; --report-every must be a positive multiple of 10".into(),
            );
        }
        Ok(Some(options))
    }
}

fn json_string(value: &str) -> String {
    let mut result = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result.push('"');
    result
}

fn json_numbers(value: Option<f64>) -> String {
    value
        .filter(|x| x.is_finite())
        .map_or_else(|| "null".into(), |x| x.to_string())
}

fn json_array(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|s| json_string(s))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn atomic_write(path: &Path, content: &str) -> io::Result<()> {
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, content)?;
    fs::rename(temporary, path)
}

fn append(path: &Path, content: &str) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{content}")?;
    file.sync_data()
}

fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

struct Lock {
    _file: File,
}

impl Lock {
    fn acquire(root: &Path) -> Result<Self> {
        let path = root.join("search.lock");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        file.try_lock()
            .map_err(|error| format!("search directory is already locked: {error}"))?;
        file.set_len(0)?;
        writeln!(file, "{}", std::process::id())?;
        Ok(Self { _file: file })
    }
}

/// Guarantees that an error in journaling or parsing cannot leave a CUDA child running.
struct TrialProcess(Child);

impl Drop for TrialProcess {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

#[derive(Clone, Debug)]
struct Candidate(Vec<String>);

impl Candidate {
    fn simple(model: &str, width: usize, depth: usize, mesh_depth: usize) -> Self {
        let mut candidate = Self(vec!["grok-simple".into()]);
        for (flag, value) in [
            ("--model", model.to_string()),
            ("--width", width.to_string()),
            ("--depth", depth.to_string()),
            ("--mesh-depth", mesh_depth.to_string()),
            ("--activation", "relu".into()),
            ("--encoding", "one-hot".into()),
            ("--connectivity", "butterfly".into()),
            ("--mesh-learning-rate-multiplier", "1".into()),
            ("--learning-rate", "0.001".into()),
            ("--weight-decay", "0.1".into()),
            ("--seed", "7".into()),
        ] {
            candidate.set(flag, value);
        }
        candidate
    }

    fn transformer(
        width: usize,
        heads: usize,
        layers: usize,
        attention: usize,
        ffn: usize,
    ) -> Self {
        let mut candidate = Self(vec!["grok".into()]);
        for (flag, value) in [
            ("--width", width.to_string()),
            ("--heads", heads.to_string()),
            ("--layers", layers.to_string()),
            ("--attention-depth", attention.to_string()),
            ("--ffn-depth", ffn.to_string()),
            ("--connectivity", "neighbor".into()),
            ("--embedding", "table".into()),
            ("--embedding-form", "parity".into()),
            ("--embedding-order", "token".into()),
            ("--embedding-norm", "none".into()),
            ("--init-gain", "1.4".into()),
            ("--embedding-output-gain", "0.1".into()),
            ("--qk-signs", "0".into()),
            ("--rope", "1".into()),
            ("--qk-norm", "1".into()),
            ("--xsa", "1".into()),
            ("--eval-batch", "64".into()),
            ("--workspace-cache", "3".into()),
            ("--learning-rate", "0.001".into()),
            ("--weight-decay", "0.1".into()),
            ("--seed", "7".into()),
        ] {
            candidate.set(flag, value);
        }
        candidate
    }

    fn get(&self, flag: &str) -> Option<&str> {
        self.0
            .iter()
            .position(|s| s == flag)
            .and_then(|i| self.0.get(i + 1))
            .map(String::as_str)
    }

    fn set(&mut self, flag: &str, value: impl ToString) {
        let value = value.to_string();
        if let Some(index) = self.0.iter().position(|s| s == flag) {
            self.0[index + 1] = value;
        } else {
            self.0.extend([flag.to_string(), value]);
        }
    }

    fn switch(&mut self, flag: &str, enabled: bool) {
        self.0.retain(|s| s != flag);
        if enabled {
            self.0.push(flag.into());
        }
    }

    fn key(&self) -> String {
        // Canonical order makes mutations and restarts deduplicate the same configuration.
        let mut entries = Vec::new();
        let mut i = 1;
        while i < self.0.len() {
            let flag = &self.0[i];
            if flag == "--tied-mesh" || flag == "--embedding-init-unit" {
                entries.push(format!("{flag}=true"));
                i += 1;
            } else {
                entries.push(format!("{flag}={}", self.0[i + 1]));
                i += 2;
            }
        }
        entries.sort();
        format!("{}|{}", self.0[0], entries.join("|"))
    }

    fn label(&self) -> String {
        format!(
            "{}-w{}-d{}-s{}",
            self.get("--model").unwrap_or("mesh-gpt"),
            self.get("--width").unwrap(),
            self.get("--depth")
                .or_else(|| self.get("--layers"))
                .unwrap(),
            self.get("--seed").unwrap()
        )
    }

    fn command(&self, policy: &Policy, output: &Path) -> Vec<String> {
        let mut arguments = self.0.clone();
        for (flag, value) in [
            ("--operation", "div".into()),
            ("--modulus", "97".into()),
            ("--train-fraction", "0.5".into()),
            ("--batch", "512".into()),
            ("--report-every", policy.report_every.to_string()),
            ("--epochs", policy.epochs.to_string()),
            (
                "--stop-validation-accuracy",
                policy.final_accuracy.to_string(),
            ),
            ("--output", output.to_string_lossy().into_owned()),
        ] {
            arguments.extend([flag.into(), value]);
        }
        arguments
    }
}

fn read_queue(path: &Path) -> Result<Vec<Candidate>> {
    let mut candidates = Vec::new();
    for (line_number, line) in fs::read_to_string(path)?.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let arguments: Vec<String> = line.split_whitespace().map(String::from).collect();
        let mut i = 1;
        while i < arguments.len() {
            let flag = &arguments[i];
            if flag == "--tied-mesh" || flag == "--embedding-init-unit" {
                i += 1;
                continue;
            }
            if ![
                "--model",
                "--width",
                "--depth",
                "--mesh-depth",
                "--activation",
                "--encoding",
                "--connectivity",
                "--mesh-learning-rate-multiplier",
                "--learning-rate",
                "--weight-decay",
                "--seed",
                "--heads",
                "--layers",
                "--attention-depth",
                "--ffn-depth",
                "--embedding",
                "--embedding-form",
                "--embedding-order",
                "--embedding-norm",
                "--init-gain",
                "--embedding-output-gain",
                "--qk-signs",
                "--rope",
                "--qk-norm",
                "--xsa",
                "--eval-batch",
                "--workspace-cache",
            ]
            .contains(&flag.as_str())
                || i + 1 == arguments.len()
                || arguments[i + 1].starts_with("--")
            {
                return Err(
                    format!("invalid queue option at line {}: {flag}", line_number + 1).into(),
                );
            }
            i += 2;
        }
        let candidate = Candidate(arguments);
        let command = candidate.0.first().map(String::as_str);
        let mesh = command == Some("grok")
            || (command == Some("grok-simple")
                && [
                    "mesh-mlp",
                    "residual-mesh-mlp",
                    "tied-concat-mesh-mlp",
                    "tied-concat-residual-mesh-mlp",
                    "tied-rotary-residual-mesh-mlp",
                ]
                .contains(&candidate.get("--model").unwrap_or("")));
        if !mesh
            || candidate.get("--width").is_none()
            || candidate.get("--seed").is_none()
            || candidate
                .get("--depth")
                .or_else(|| candidate.get("--layers"))
                .is_none()
            || candidate
                .get("--encoding")
                .is_some_and(|value| value != "one-hot")
            || candidate
                .get("--embedding-order")
                .is_some_and(|value| value != "token")
        {
            return Err(format!("queue line {} must specify a mesh model, width, depth/layers, seed, and unassisted inputs", line_number + 1).into());
        }
        candidates.push(candidate);
    }
    if candidates.is_empty() {
        return Err("queue contains no candidates".into());
    }
    Ok(candidates)
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn choose<T: Copy>(&mut self, values: &[T]) -> T {
        values[(self.next() % values.len() as u64) as usize]
    }
}

fn proposal(index: u64, search_seed: u64, best: Option<&Candidate>) -> Candidate {
    // Interleave new hypotheses rather than exhaust one architecture first.
    if index < 12 {
        let mut c = match index {
            0 | 3 | 6 | 11 => Candidate::simple(
                "residual-mesh-mlp",
                match index {
                    3 => 256,
                    11 => 64,
                    _ => 128,
                },
                2,
                if index == 6 { 3 } else { 2 },
            ),
            1 | 4 | 7 | 10 => Candidate::transformer(
                if index == 4 {
                    256
                } else if index == 10 {
                    64
                } else {
                    128
                },
                if index == 10 { 2 } else { 4 },
                2,
                if index == 10 { 4 } else { 8 },
                if index == 10 { 8 } else { 16 },
            ),
            2 => Candidate::simple("mesh-mlp", 128, 5, 4),
            5 => Candidate::simple("residual-mesh-mlp", 128, 2, 32),
            8 => Candidate::simple("mesh-mlp", 64, 17, 4),
            9 => Candidate::simple("residual-mesh-mlp", 256, 2, 16),
            _ => unreachable!(),
        };
        match index {
            0 | 3 | 6 | 11 => c.set("--activation", "square"),
            1 => {
                c.switch("--embedding-init-unit", true);
                c.set("--weight-decay", "0.01");
            }
            2 => {
                c.set("--connectivity", "own");
                c.switch("--tied-mesh", true);
                c.set("--weight-decay", "0.03");
                c.set("--mesh-learning-rate-multiplier", "2");
            }
            7 => {
                c.set("--init-gain", "1");
                c.set("--embedding-output-gain", "1");
            }
            8 => {
                c.set("--connectivity", "neighbor");
                c.set("--weight-decay", "0.3");
                c.set("--mesh-learning-rate-multiplier", "2");
            }
            10 => {
                c.set("--init-gain", "1");
                c.set("--embedding-output-gain", "0.3");
                c.set("--weight-decay", "0.03");
            }
            _ => {}
        }
        return c;
    }
    let mut random = Random(search_seed ^ index.wrapping_mul(0xd1342543de82ef95));
    if index.is_multiple_of(4)
        && let Some(best) = best
    {
        let mut c = best.clone();
        match random.choose(&[0, 1, 2, 3]) {
            0 => c.set(
                "--weight-decay",
                random.choose(&["0.005", "0.01", "0.03", "0.1", "0.3", "1"]),
            ),
            1 => c.set(
                "--learning-rate",
                random.choose(&["0.0003", "0.001", "0.003"]),
            ),
            2 => c.set(
                "--seed",
                random.choose(&[7, 8, 9, 10, 11, 12, 13, 17, 23, 42]),
            ),
            _ => c.set("--connectivity", random.choose(&["neighbor", "butterfly"])),
        }
        return c;
    }
    let family = random.choose(&[0, 1, 2, 3]);
    let mut c = if family == 3 {
        let width = random.choose(&[32, 64, 128, 256, 512]);
        let mut c = Candidate::transformer(
            width,
            random.choose(&[1, 2, 4, 8]),
            random.choose(&[1, 2, 4]),
            random.choose(&[4, 8, 16]),
            random.choose(&[8, 16, 32]),
        );
        c.set(
            "--embedding",
            random.choose(&["table", "table", "table", "mesh"]),
        );
        c.set(
            "--embedding-norm",
            random.choose(&["none", "none", "none", "unit", "rms"]),
        );
        c.set("--init-gain", random.choose(&["0.7", "1", "1.4", "1.8"]));
        c.set(
            "--embedding-output-gain",
            random.choose(&["0.01", "0.1", "0.3", "1"]),
        );
        c.set("--qk-signs", random.choose(&["0", "1"]));
        c.set("--qk-norm", random.choose(&["1", "1", "1", "0"]));
        c.set("--xsa", random.choose(&["1", "1", "0"]));
        let initial_unit = random.choose(&[true, false])
            && c.get("--embedding") == Some("table")
            && c.get("--embedding-norm") == Some("none");
        c.switch("--embedding-init-unit", initial_unit);
        c
    } else {
        let width = random.choose(&[32, 48, 64, 96, 128, 192, 256, 384, 512]);
        if family == 0 {
            let mut c = Candidate::simple(
                "residual-mesh-mlp",
                width,
                random.choose(&[1, 2, 4]),
                random.choose(&[1, 2, 3, 4]),
            );
            c.set("--activation", "square");
            c
        } else if family == 1 {
            let depth = random.choose(&[1, 2, 4]);
            Candidate::simple(
                "residual-mesh-mlp",
                width,
                depth,
                random
                    .choose(&[2, 4, 8, 16, 24, 32, 48, 64])
                    .min(64 / depth),
            )
        } else {
            let mut c = Candidate::simple(
                random.choose(&["mesh-mlp", "mesh-mlp", "tied-concat-mesh-mlp"]),
                width,
                random.choose(&[2, 3, 5, 9, 17, 33]),
                4,
            );
            c.switch("--tied-mesh", random.choose(&[true, false]));
            c
        }
    };
    c.set(
        "--connectivity",
        random.choose(if family == 3 {
            &["neighbor", "butterfly"][..]
        } else {
            &["own", "neighbor", "butterfly"][..]
        }),
    );
    if c.0[0] == "grok-simple" {
        c.set(
            "--mesh-learning-rate-multiplier",
            random.choose(&["1", "2", "4", "8", "16"]),
        );
    }
    c.set(
        "--weight-decay",
        random.choose(&["0.005", "0.01", "0.03", "0.1", "0.3", "1"]),
    );
    c.set(
        "--learning-rate",
        random.choose(&["0.0003", "0.001", "0.003"]),
    );
    c.set(
        "--seed",
        random.choose(&[7, 8, 9, 10, 11, 12, 13, 17, 23, 42]),
    );
    c
}

#[derive(Clone, Copy, Debug)]
struct Row {
    epoch: f64,
    train: f64,
    test: f64,
}

#[derive(Default, Debug)]
struct Stats {
    rows: usize,
    last: Option<Row>,
    active_plateau: Option<f64>,
    plateau_start: Option<f64>,
    plateau_end: Option<f64>,
    plateau_epochs: f64,
    last_low: Option<f64>,
    low_before_90: Option<f64>,
    first_90: Option<f64>,
    first_99: Option<f64>,
    plateau_before_90: f64,
}

impl Stats {
    fn observe(&mut self, row: Row, policy: &Policy) {
        self.rows += 1;
        self.last = Some(row);
        if self.first_90.is_none() {
            if row.test <= policy.low_accuracy {
                self.last_low = Some(row.epoch);
            }
            if row.train >= policy.train_accuracy && row.test <= policy.low_accuracy {
                let start = *self.active_plateau.get_or_insert(row.epoch);
                let duration = row.epoch - start;
                if duration >= self.plateau_epochs {
                    self.plateau_epochs = duration;
                    self.plateau_start = Some(start);
                    self.plateau_end = Some(row.epoch);
                }
            } else {
                self.active_plateau = None;
            }
            if row.test >= policy.high_accuracy {
                self.first_90 = Some(row.epoch);
                self.low_before_90 = self.last_low;
                self.plateau_before_90 = self.plateau_epochs;
            }
        }
        if row.test >= policy.final_accuracy && self.first_99.is_none() {
            self.first_99 = Some(row.epoch);
        }
    }

    fn rise(&self) -> Option<f64> {
        self.first_90
            .zip(self.low_before_90)
            .map(|(high, low)| high - low)
    }

    fn matched(&self, policy: &Policy) -> bool {
        self.plateau_before_90 >= policy.plateau_epochs
            && self
                .rise()
                .is_some_and(|epochs| epochs <= policy.jump_epochs)
            && self.first_99.is_some()
    }

    fn score(&self, policy: &Policy) -> f64 {
        // Prefer measured delayed transitions over underfit plateaus. Unconfirmed runs
        // can guide only when no completed generalizing curve has been observed yet.
        if self.first_99.is_some() && self.plateau_before_90 > 0.0 {
            (if self.matched(policy) {
                1_000_000.0
            } else {
                1_000.0
            }) + (self.plateau_before_90 / policy.plateau_epochs).min(1.0) * 10.0
                + (policy.jump_epochs
                    / self
                        .rise()
                        .unwrap_or(1e9)
                        .max(policy.report_every as f64 / 10.0))
                .min(10.0)
                    * 5.0
        } else if self.plateau_epochs > 0.0 {
            (self.plateau_epochs / policy.plateau_epochs).min(5.0)
                + self.last.map_or(0.0, |r| r.test)
        } else {
            -1.0
        }
    }

    fn json(&self, policy: &Policy) -> String {
        format!(
            "{{\"rows\":{},\"epoch\":{},\"train_accuracy\":{},\"test_accuracy\":{},\"plateau_start_epoch\":{},\"plateau_end_epoch\":{},\"plateau_epochs\":{},\"plateau_before_90_epochs\":{},\"last_low_before_90_epoch\":{},\"first_test_90_epoch\":{},\"rise_epochs\":{},\"first_test_99_epoch\":{},\"matched\":{}}}",
            self.rows,
            json_numbers(self.last.map(|r| r.epoch)),
            json_numbers(self.last.map(|r| r.train)),
            json_numbers(self.last.map(|r| r.test)),
            json_numbers(self.plateau_start),
            json_numbers(self.plateau_end),
            self.plateau_epochs,
            self.plateau_before_90,
            json_numbers(self.low_before_90),
            json_numbers(self.first_90),
            json_numbers(self.rise()),
            json_numbers(self.first_99),
            self.matched(policy),
        )
    }
}

fn scan_curve(contents: &str, policy: &Policy) -> Result<(Stats, Option<String>)> {
    let mut stats = Stats::default();
    // A live writer may have emitted only part of the final line.
    let complete = &contents[..contents.rfind('\n').map_or(0, |i| i + 1)];
    let mut lines = complete.lines();
    let Some(header) = lines.next() else {
        return Ok((stats, None));
    };
    let columns: Vec<_> = header.split(',').collect();
    let index = |name: &str| -> Result<usize> {
        columns
            .iter()
            .position(|s| *s == name)
            .ok_or_else(|| format!("missing CSV column {name}").into())
    };
    let epoch = index("completed_epochs")?;
    let train = index("train_accuracy")?;
    let test = index("validation_accuracy")?;
    let train_loss = index("train_cross_entropy")?;
    let test_loss = index("validation_cross_entropy")?;
    for (number, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split(',').collect();
        if fields.len() != columns.len() {
            return Err(format!("malformed CSV row {}", number + 2).into());
        }
        let values = [epoch, train, test, train_loss, test_loss].map(|i| fields[i].parse::<f64>());
        let mut parsed = [0.0; 5];
        for (destination, value) in parsed.iter_mut().zip(values) {
            *destination = value?;
        }
        if parsed.iter().any(|x| !x.is_finite())
            || !(0.0..=1.0).contains(&parsed[1])
            || !(0.0..=1.0).contains(&parsed[2])
            || parsed[3] < -1e-6
            || parsed[4] < -1e-6
        {
            return Ok((
                stats,
                Some(format!(
                    "invalid/nonfinite metric at CSV row {}",
                    number + 2
                )),
            ));
        }
        let row = Row {
            epoch: parsed[0],
            train: parsed[1],
            test: parsed[2],
        };
        if row.epoch < 0.0 || stats.last.is_some_and(|last| row.epoch <= last.epoch) {
            return Err("CSV epoch order must strictly increase".into());
        }
        // Missing reports break a contiguous plateau rather than bridge unobserved time.
        if stats
            .last
            .is_some_and(|last| row.epoch - last.epoch > policy.report_every as f64 / 10.0 + 1e-6)
        {
            stats.active_plateau = None;
        }
        stats.observe(row, policy);
    }
    Ok((stats, None))
}

fn sha256(path: &Path) -> Result<String> {
    let output = Command::new("sha256sum").arg(path).output()?;
    if !output.status.success() {
        return Err(format!("sha256sum failed for {}", path.display()).into());
    }
    let digest = String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string();
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid SHA256 output".into());
    }
    Ok(digest)
}

fn snapshot_sources(source: &Path, destination: &Path) -> io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            snapshot_sources(&entry.path(), &destination.join(entry.file_name()))?;
        } else if kind.is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "rs")
        {
            fs::copy(entry.path(), destination.join(entry.file_name()))?;
        }
    }
    Ok(())
}

fn freeze(root: &Path, source: &Path, policy: &Policy) -> Result<PathBuf> {
    let bin = root.join("bin");
    fs::create_dir_all(&bin)?;
    let trainer = bin.join("mesh-gpt");
    if !trainer.exists() {
        fs::copy(source, &trainer)?;
    }
    let digest = sha256(&trainer)?;
    let contract = format!("{}\ntrainer_sha256={digest}\n", policy.json());
    let path = root.join("policy.txt");
    if path.exists() {
        if fs::read_to_string(path)? != contract {
            return Err(
                "existing search has a different policy or trainer hash; use a new --output".into(),
            );
        }
    } else {
        let sources = root.join("sources");
        snapshot_sources(Path::new("src"), &sources.join("src"))?;
        for path in [
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo/config.toml",
        ] {
            let destination = sources.join(path);
            fs::create_dir_all(destination.parent().unwrap())?;
            fs::copy(path, destination)?;
        }
        fs::copy(std::env::current_exe()?, bin.join("sharp-mesh-search"))?;
        let git = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
        fs::write(root.join("source-head.txt"), git.stdout)?;
        fs::write(
            root.join("runner-sha256.txt"),
            sha256(&bin.join("sharp-mesh-search"))?,
        )?;
        atomic_write(&path, &contract)?;
    }
    // The CUDA binary stays frozen across restarts. Record the actual host scheduler
    // on each invocation as well, so a scheduler fix does not erase its provenance.
    let session = root
        .join("sessions")
        .join(format!("{}-{}", timestamp(), std::process::id()));
    fs::create_dir_all(&session)?;
    fs::copy(std::env::current_exe()?, session.join("sharp-mesh-search"))?;
    fs::copy(
        "src/bin/sharp-mesh-search.rs",
        session.join("sharp-mesh-search.rs"),
    )?;
    fs::write(
        session.join("runner-sha256.txt"),
        sha256(&session.join("sharp-mesh-search"))?,
    )?;
    Ok(trainer)
}

struct Journal {
    completed: usize,
    matches: usize,
    seen: HashSet<String>,
    best: Option<Candidate>,
    best_score: f64,
}

impl Journal {
    fn load(root: &Path) -> Result<Self> {
        let mut journal = Self {
            completed: 0,
            matches: 0,
            seen: HashSet::new(),
            best: None,
            best_score: -1.0,
        };
        let path = root.join("finished.keys");
        if path.exists() {
            for line in fs::read_to_string(path)?.lines() {
                journal.seen.insert(line.into());
                journal.completed += 1;
            }
        }
        let matches = root.join("matches.jsonl");
        if matches.exists() {
            journal.matches = fs::read_to_string(matches)?.lines().count();
        }
        let best = root.join("best.args");
        if best.exists() {
            journal.best = Some(Candidate(
                fs::read_to_string(best)?
                    .lines()
                    .map(String::from)
                    .collect(),
            ));
            journal.best_score = fs::read_to_string(root.join("best.score"))?
                .trim()
                .parse()?;
        }
        Ok(journal)
    }

    fn state(
        &self,
        root: &Path,
        status: &str,
        trial: Option<(&str, u32, &Stats)>,
        policy: &Policy,
    ) -> io::Result<()> {
        let active = trial.map_or_else(
            || "null".into(),
            |(label, pid, stats)| {
                format!(
                    "{{\"label\":{},\"trainer_pid\":{pid},\"directory\":{},\"statistics\":{}}}",
                    json_string(label),
                    json_string(&root.join("trials").join(label).to_string_lossy()),
                    stats.json(policy)
                )
            },
        );
        atomic_write(
            &root.join("state.json"),
            &format!(
                "{{\"status\":{},\"pid\":{},\"updated_unix_seconds\":{},\"completed_trials\":{},\"matches\":{},\"policy\":{},\"active_trial\":{active}}}\n",
                json_string(status),
                std::process::id(),
                timestamp(),
                self.completed,
                self.matches,
                policy.json()
            ),
        )
    }
}

fn trial(
    root: &Path,
    trainer: &Path,
    candidate: &Candidate,
    label: &str,
    journal: &Journal,
    policy: &Policy,
) -> Result<(Stats, String)> {
    let directory = root.join("trials").join(label);
    fs::create_dir(&directory)?;
    let arguments = candidate.command(policy, &directory);
    fs::write(directory.join("arguments.json"), json_array(&arguments))?;
    fs::write(
        directory.join("candidate.args"),
        candidate.0.join("\n") + "\n",
    )?;
    let log = File::create(directory.join("console.log"))?;
    let mut child = TrialProcess(
        Command::new(trainer)
            .args(&arguments)
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log)
            .spawn()?,
    );
    println!("START {label}: {}", json_array(&candidate.0));
    let started = Instant::now();
    let history = directory.join("loss.csv");
    let mut last_size = u64::MAX;
    let mut stats = Stats::default();
    let mut detail = None;
    let outcome = loop {
        let exit = child.0.try_wait()?;
        let size = match fs::metadata(&history) {
            Ok(metadata) => metadata.len(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error.into()),
        };
        if size != last_size || exit.is_some() {
            last_size = size;
            if size > 0 {
                let (updated, divergence) = scan_curve(&fs::read_to_string(&history)?, policy)?;
                stats = updated;
                detail = divergence;
            }
            journal.state(root, "running", Some((label, child.0.id(), &stats)), policy)?;
        }
        if root.join("STOP").exists() {
            break "interrupted".to_string();
        }
        if detail.is_some() {
            break "diverged".to_string();
        }
        if let Some(exit) = exit {
            if exit.success() {
                let metadata = fs::read_to_string(directory.join("run.txt"))?;
                if !metadata.lines().any(|line| line == "status=complete") || stats.rows < 2 {
                    return Err("trainer exited without complete metadata and metrics".into());
                }
                break "complete".to_string();
            }
            detail = Some(format!("trainer exit {exit}"));
            break "failed".to_string();
        }
        if started.elapsed().as_secs() >= policy.timeout_seconds {
            detail = Some("wall-time limit; this is not a full-epoch-budget result".into());
            break "timeout".to_string();
        }
        thread::sleep(Duration::from_millis(250));
    };
    drop(child);
    fs::write(
        directory.join("outcome.json"),
        format!(
            "{{\"status\":{},\"detail\":{},\"elapsed_seconds\":{},\"statistics\":{}}}\n",
            json_string(&outcome),
            detail.as_deref().map_or_else(|| "null".into(), json_string),
            started.elapsed().as_secs_f64(),
            stats.json(policy),
        ),
    )?;
    Ok((stats, outcome))
}

fn search(mut options: Options) -> Result<()> {
    let queue = match &options.queue {
        Some(path) => {
            options.policy.queue_sha256 = Some(sha256(path)?);
            read_queue(path)?
        }
        None => Vec::new(),
    };
    let propose = |index: u64, best: Option<&Candidate>| {
        queue.get(index as usize).cloned().unwrap_or_else(|| {
            proposal(index - queue.len() as u64, options.policy.search_seed, best)
        })
    };
    if options.dry_run > 0 {
        println!("POLICY {}", options.policy.json());
        for index in 0..options.dry_run {
            let candidate = propose(index, None);
            println!(
                "{} {}",
                index + 1,
                json_array(
                    &candidate.command(&options.policy, &options.output.join(candidate.label()))
                )
            );
        }
        return Ok(());
    }
    fs::create_dir_all(&options.output)?;
    let root = fs::canonicalize(&options.output)?;
    let _lock = Lock::acquire(&root)?;
    let trainer = freeze(&root, &options.trainer, &options.policy)?;
    if let Some(path) = &options.queue {
        fs::copy(path, root.join("queue.args"))?;
    }
    fs::create_dir_all(root.join("trials"))?;
    let mut journal = Journal::load(&root)?;
    let cursor_path = root.join("cursor");
    let mut index = if cursor_path.exists() {
        fs::read_to_string(&cursor_path)?.trim().parse()?
    } else {
        0
    };
    let mut new_trials = 0;
    let mut failures = 0;
    println!(
        "SEARCH pid={} output={} policy={}",
        std::process::id(),
        root.display(),
        options.policy.json()
    );
    journal.state(&root, "starting", None, &options.policy)?;
    loop {
        if root.join("STOP").exists() {
            journal.state(&root, "stopped", None, &options.policy)?;
            println!("STOP file observed; all owned trainer processes have exited");
            return Ok(());
        }
        if (options.max_trials > 0 && new_trials >= options.max_trials)
            || (options.stop_after_matches > 0
                && journal.matches as u64 >= options.stop_after_matches)
        {
            journal.state(&root, "limit_reached", None, &options.policy)?;
            return Ok(());
        }
        let candidate = propose(index, journal.best.as_ref());
        let key = candidate.key();
        if journal.seen.contains(&key) {
            index += 1;
            atomic_write(&cursor_path, &format!("{index}\n"))?;
            continue;
        }
        let mut attempt = 1;
        let label = loop {
            let label = format!("{:06}-{}-a{attempt:02}", index + 1, candidate.label());
            if !root.join("trials").join(&label).exists() {
                break label;
            }
            attempt += 1;
        };
        let (stats, outcome) = match trial(
            &root,
            &trainer,
            &candidate,
            &label,
            &journal,
            &options.policy,
        ) {
            Ok(result) => result,
            Err(error) => {
                journal.state(&root, "error", None, &options.policy)?;
                return Err(error);
            }
        };
        let matched = outcome == "complete" && stats.matched(&options.policy);
        let record = format!(
            "{{\"trial\":{},\"proposal_index\":{index},\"status\":{},\"matched\":{matched},\"directory\":{},\"candidate\":{},\"statistics\":{}}}",
            json_string(&label),
            json_string(&outcome),
            json_string(&root.join("trials").join(&label).to_string_lossy()),
            json_array(&candidate.0),
            stats.json(&options.policy),
        );
        append(&root.join("results.jsonl"), &record)?;
        println!("END {label}: {outcome} {}", stats.json(&options.policy));
        if outcome == "interrupted" {
            continue;
        }
        if matched {
            append(&root.join("matches.jsonl"), &record)?;
            journal.matches += 1;
            println!("MATCH {label}: long plateau followed by sharp generalization");
        }
        let score = stats.score(&options.policy);
        if outcome == "complete" && score > journal.best_score {
            atomic_write(&root.join("best.args"), &(candidate.0.join("\n") + "\n"))?;
            atomic_write(&root.join("best.score"), &score.to_string())?;
            atomic_write(&root.join("best.json"), &record)?;
            journal.best = Some(candidate);
            journal.best_score = score;
        }
        append(&root.join("finished.keys"), &key)?;
        journal.seen.insert(key);
        journal.completed += 1;
        new_trials += 1;
        index += 1;
        atomic_write(&cursor_path, &format!("{index}\n"))?;
        failures = if outcome == "failed" { failures + 1 } else { 0 };
        if failures >= 3 {
            journal.state(&root, "error", None, &options.policy)?;
            return Err("three consecutive trainer failures; inspect the trial console logs before restarting".into());
        }
    }
}

fn main() {
    let result = Options::parse().and_then(|options| match options {
        Some(options) => search(options),
        None => Ok(()),
    });
    if let Err(error) = result {
        eprintln!("sharp-mesh-search: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    struct Temporary(PathBuf);

    impl Temporary {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("sharp-mesh-search-{label}-{}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn row(epoch: f64, train: f64, test: f64) -> Row {
        Row { epoch, train, test }
    }

    #[test]
    fn delayed_sharp_curve_matches_but_direct_and_gradual_generalization_do_not() {
        let policy = Policy::default();
        let mut delayed = Stats::default();
        for epoch in (100..=1_200).step_by(25) {
            delayed.observe(row(epoch as f64, 1.0, 0.02), &policy);
        }
        delayed.observe(row(1_325.0, 1.0, 0.91), &policy);
        assert!(!delayed.matched(&policy));
        delayed.observe(row(1_400.0, 1.0, 0.995), &policy);
        assert!(delayed.matched(&policy));
        assert_eq!(delayed.rise(), Some(125.0));
        let mut direct = Stats::default();
        direct.observe(row(25.0, 0.5, 0.05), &policy);
        direct.observe(row(50.0, 1.0, 0.995), &policy);
        assert!(!direct.matched(&policy));
        delayed.first_90 = Some(2_000.0);
        assert!(!delayed.matched(&policy));
    }

    #[test]
    fn train_accuracy_dips_reset_the_plateau_and_later_dips_do_not_rewrite_first_transition() {
        let policy = Policy::default();
        let mut stats = Stats::default();
        for epoch in (0..=500).step_by(25) {
            stats.observe(row(epoch as f64, 1.0, 0.02), &policy);
        }
        stats.observe(row(525.0, 0.98, 0.02), &policy);
        for epoch in (550..=1_050).step_by(25) {
            stats.observe(row(epoch as f64, 1.0, 0.02), &policy);
        }
        assert_eq!(stats.plateau_epochs, 500.0);
        stats.observe(row(1_100.0, 1.0, 0.91), &policy);
        stats.observe(row(1_150.0, 1.0, 0.995), &policy);
        for epoch in (1_200..=3_000).step_by(25) {
            stats.observe(row(epoch as f64, 1.0, 0.02), &policy);
        }
        stats.observe(row(3_100.0, 1.0, 0.995), &policy);
        assert!(!stats.matched(&policy));
        assert_eq!(stats.first_90, Some(1_100.0));
        assert_eq!(stats.low_before_90, Some(1_050.0));
    }

    #[test]
    fn csv_reader_ignores_partial_lines_detects_divergence_and_breaks_missing_reports() {
        let policy = Policy::default();
        let header = "completed_epochs,train_accuracy,validation_accuracy,train_cross_entropy,validation_cross_entropy\n";
        let csv = format!(
            "{header}0,1,0.02,0.001,5\n25,1,0.02,0.001,5\n1000,1,0.02,0.001,5\n1025,1,0.02,"
        );
        let (stats, divergence) = scan_curve(&csv, &policy).unwrap();
        assert_eq!(stats.rows, 3);
        assert_eq!(stats.plateau_epochs, 25.0);
        assert!(divergence.is_none());
        let (stats, divergence) =
            scan_curve(&format!("{header}0,1,0.02,NaN,5\n"), &policy).unwrap();
        assert_eq!(stats.rows, 0);
        assert!(divergence.is_some());
        assert!(scan_curve(&format!("{header}0,1,0.02\n"), &policy).is_err());
    }

    #[test]
    fn proposals_are_reproducible_mesh_only_and_use_full_epoch_budgets() {
        let policy = Policy::default();
        let mut seen = HashSet::new();
        for index in 0..500 {
            let c = proposal(index, policy.search_seed, None);
            assert_eq!(c.key(), proposal(index, policy.search_seed, None).key());
            if c.0[0] == "grok-simple" {
                assert!(
                    ["mesh-mlp", "residual-mesh-mlp", "tied-concat-mesh-mlp"]
                        .contains(&c.get("--model").unwrap())
                );
                assert_eq!(c.get("--encoding"), Some("one-hot"));
                let config = grokking_abmod::simple::SimpleConfig {
                    architecture: c.get("--model").unwrap().parse().unwrap(),
                    activation: c.get("--activation").unwrap().parse().unwrap(),
                    encoding: c.get("--encoding").unwrap().parse().unwrap(),
                    width: c.get("--width").unwrap().parse().unwrap(),
                    depth: c.get("--depth").unwrap().parse().unwrap(),
                    mesh_depth: c.get("--mesh-depth").unwrap().parse().unwrap(),
                    connectivity: c.get("--connectivity").unwrap().parse().unwrap(),
                    tied_mesh: c.0.iter().any(|s| s == "--tied-mesh"),
                    mesh_learning_rate_multiplier: c
                        .get("--mesh-learning-rate-multiplier")
                        .unwrap()
                        .parse()
                        .unwrap(),
                    ..Default::default()
                };
                config.validate().unwrap();
                if c.get("--model") == Some("residual-mesh-mlp") {
                    assert!(!c.0.iter().any(|s| s == "--tied-mesh"));
                }
            } else {
                let width: usize = c.get("--width").unwrap().parse().unwrap();
                let heads: usize = c.get("--heads").unwrap().parse().unwrap();
                assert_eq!(width % heads, 0);
                assert_eq!(c.get("--embedding-order"), Some("token"));
                if c.0.iter().any(|s| s == "--embedding-init-unit") {
                    assert_eq!(c.get("--embedding"), Some("table"));
                    assert_eq!(c.get("--embedding-norm"), Some("none"));
                }
                grokking_abmod::gpt::GptConfig {
                    vocab_size: 100,
                    context: 4,
                    width,
                    heads,
                    layers: c.get("--layers").unwrap().parse().unwrap(),
                    attention_mesh_depth: c.get("--attention-depth").unwrap().parse().unwrap(),
                    ffn_mesh_depth: c.get("--ffn-depth").unwrap().parse().unwrap(),
                    embedding: c.get("--embedding").unwrap().parse().unwrap(),
                    connectivity: c.get("--connectivity").unwrap().parse().unwrap(),
                    ..Default::default()
                }
                .validate()
                .unwrap();
            }
            let command = c.command(&policy, Path::new("/tmp/trial"));
            assert!(command.windows(2).any(|pair| pair == ["--epochs", "10000"]));
            seen.insert(c.key());
        }
        assert!(seen.len() > 450);
    }

    #[test]
    fn candidate_key_is_independent_of_option_order_and_mutating_to_same_value() {
        let mut a = Candidate::simple("mesh-mlp", 64, 5, 4);
        let key = a.key();
        a.set("--width", 64);
        a.switch("--tied-mesh", true);
        a.switch("--tied-mesh", false);
        assert_eq!(a.key(), key);
        let mut b = Candidate(vec!["grok-simple".into()]);
        for pair in a.0[1..].as_chunks::<2>().0.iter().rev() {
            b.set(&pair[0], &pair[1]);
        }
        assert_eq!(b.key(), key);
    }

    #[test]
    fn directory_lock_prevents_parallel_workers_and_is_released_on_drop() {
        let directory = Temporary::new("lock");
        let lock = Lock::acquire(&directory.0).unwrap();
        assert!(Lock::acquire(&directory.0).is_err());
        drop(lock);
        assert!(Lock::acquire(&directory.0).is_ok());
    }

    #[test]
    fn fine_reports_confirm_short_plateaus_without_bridging_missing_epochs() {
        let policy = Policy {
            plateau_epochs: 25.0,
            report_every: 10,
            ..Default::default()
        };
        let header = "completed_epochs,train_accuracy,validation_accuracy,train_cross_entropy,validation_cross_entropy\n";
        let mut curve = header.to_string();
        for epoch in 0..=25 {
            curve.push_str(&format!("{epoch},1,0.02,0.001,5\n"));
        }
        curve.push_str("26,1,0.91,0.001,0.3\n27,1,0.995,0.001,0.01\n");
        let (stats, _) = scan_curve(&curve, &policy).unwrap();
        assert!(stats.matched(&policy));
        assert_eq!(stats.plateau_epochs, 25.0);
        assert_eq!(stats.rise(), Some(1.0));
        let missing = curve.replace("13,1,0.02,0.001,5\n", "");
        let (stats, _) = scan_curve(&missing, &policy).unwrap();
        assert!(!stats.matched(&policy));
        assert_eq!(stats.plateau_epochs, 12.0);
    }

    #[test]
    fn queue_accepts_mesh_arguments_and_rejects_nonmesh_cheat_inputs_and_output_overrides() {
        let directory = Temporary::new("queue");
        let path = directory.0.join("study.args");
        let candidate = Candidate::simple("residual-mesh-mlp", 128, 2, 3);
        let line = candidate.0.join(" ");
        fs::write(&path, format!("# Research queue\n\n{line}\n")).unwrap();
        assert_eq!(read_queue(&path).unwrap()[0].key(), candidate.key());
        for invalid in [
            line.replace("residual-mesh-mlp", "residual-mlp"),
            line.replace("one-hot", "discrete-log"),
            format!("{line} --output /tmp/overwrite"),
            format!("{line} --width"),
            "grok-simple --model mesh-mlp".into(),
        ] {
            fs::write(&path, invalid).unwrap();
            assert!(read_queue(&path).is_err());
        }
    }

    #[test]
    fn stop_kills_owned_child_and_resume_keeps_interrupted_artifacts() {
        let directory = Temporary::new("stop-resume");
        fs::create_dir(directory.0.join("trials")).unwrap();
        let trainer = directory.0.join("fixture-trainer");
        fs::write(&trainer, concat!(
            "#!/bin/sh\nprevious=\nfor argument do\n",
            "  if test \"$previous\" = --output; then run_directory=$argument; fi\n",
            "  previous=$argument\ndone\n",
            "printf 'completed_epochs,train_accuracy,validation_accuracy,train_cross_entropy,validation_cross_entropy\\n0,0.01,0.01,5,5\\n25,0.5,0.02,2,5\\n' > \"$run_directory/loss.csv\"\n",
            "printf 'status=complete\\n' > \"$run_directory/run.txt\"\n",
        )).unwrap();
        fs::set_permissions(&trainer, fs::Permissions::from_mode(0o755)).unwrap();
        let policy = Policy::default();
        let candidate = proposal(0, policy.search_seed, None);
        let journal = Journal::load(&directory.0).unwrap();
        fs::write(directory.0.join("STOP"), "").unwrap();
        let (_, status) = trial(
            &directory.0,
            &trainer,
            &candidate,
            "first",
            &journal,
            &policy,
        )
        .unwrap();
        assert_eq!(status, "interrupted");
        fs::remove_file(directory.0.join("STOP")).unwrap();
        let (stats, status) = trial(
            &directory.0,
            &trainer,
            &candidate,
            "second",
            &journal,
            &policy,
        )
        .unwrap();
        assert_eq!(status, "complete");
        assert_eq!(stats.rows, 2);
        assert!(directory.0.join("trials/first/outcome.json").exists());
        append(&directory.0.join("finished.keys"), &candidate.key()).unwrap();
        let resumed = Journal::load(&directory.0).unwrap();
        assert_eq!(resumed.completed, 1);
        assert!(resumed.seen.contains(&candidate.key()));
    }
}
