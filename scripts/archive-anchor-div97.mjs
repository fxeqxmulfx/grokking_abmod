import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { gzipSync } from "node:zlib";

const [destination, ...directories] = process.argv.slice(2);
if (!destination || directories.length === 0) {
  throw new Error("Usage: node scripts/archive-anchor-div97.mjs OUTPUT RUN_DIRECTORY...");
}
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const hash = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const gitHead = () => {
  const head = fs.readFileSync(".git/HEAD", "utf8").trim();
  if (!head.startsWith("ref: ")) return head;
  const ref = head.slice(5);
  const loose = path.join(".git", ref);
  if (fs.existsSync(loose)) return fs.readFileSync(loose, "utf8").trim();
  const packed = fs.readFileSync(".git/packed-refs", "utf8").split("\n")
    .find(line => line.endsWith(` ${ref}`));
  assert(packed, `Cannot resolve Git HEAD: ${ref}`);
  return packed.split(" ")[0];
};
const series = [];
const histories = [];
const rows = [];
let curveHeader;

for (const directory of directories) {
  const summary = JSON.parse(fs.readFileSync(path.join(directory, "summary.json"), "utf8"));
  assert(summary.status === "complete", `Incomplete series: ${directory}`);
  const name = path.basename(directory);
  const keys = new Set();
  for (const run of summary.runs) {
    const key = `${run.dictionary_size}-${run.seed}`;
    assert(!keys.has(key), `Duplicate run ${name}/${key}`);
    keys.add(key);
    const trainableRows = summary.input_only ? 97 : run.dictionary_size;
    assert(run.parameters === trainableRows * summary.width + summary.width ** 2,
      `Unexpected parameter count ${name}/${key}`);
    for (const split of [run.train, run.test]) {
      assert(split.rows === 4656 && split.correct >= 0 && split.correct <= split.rows,
        `Incomplete split ${name}/${key}`);
      assert(Math.abs(split.accuracy - split.correct / split.rows) < 1e-12,
        `Inconsistent accuracy ${name}/${key}`);
      for (const counts of [split.off_dictionary_operands, split.off_dictionary_answer]) {
        assert(counts.reduce((sum, x) => sum + x[0], 0) === split.correct &&
          counts.reduce((sum, x) => sum + x[1], 0) === split.rows,
        `Incomplete strata ${name}/${key}`);
      }
    }
    const checkpoint = fs.readFileSync(path.join(directory, `k${run.dictionary_size}-seed${run.seed}.f32`));
    assert(checkpoint.length === run.parameters * 4, `Checkpoint shape ${name}/${key}`);
    run.checkpoint_sha256 = crypto.createHash("sha256").update(checkpoint).digest("hex");
    for (let offset = 0; offset < checkpoint.length; offset += 4) {
      assert(Number.isFinite(checkpoint.readFloatLE(offset)), `Nonfinite checkpoint ${name}/${key}`);
    }
    const lines = fs.readFileSync(path.join(directory, `k${run.dictionary_size}-seed${run.seed}.csv`), "utf8")
      .trim().split("\n");
    const header = lines.shift();
    curveHeader ??= header;
    assert(header === curveHeader, `Different curve columns ${name}/${key}`);
    const names = header.split(",");
    const records = lines.map((line) => {
      const values = line.split(",").map(Number);
      assert(values.length === names.length && values.every(Number.isFinite),
        `Invalid curve values ${name}/${key}`);
      return Object.fromEntries(names.map((n, i) => [n, values[i]]));
    });
    assert(records[0].epoch === 0 && records.at(-1).epoch === run.epochs,
      `Incomplete curve ${name}/${key}`);
    for (let i = 0; i < records.length; i++) {
      assert(records[i].updates === 10 * records[i].epoch,
        `Unexpected epoch definition ${name}/${key}`);
      if (i) assert(records[i].epoch > records[i - 1].epoch, `Unordered curve ${name}/${key}`);
    }
    const last = records.at(-1);
    assert(last.train_accuracy === run.train.accuracy && last.test_accuracy === run.test.accuracy,
      `Final curve differs from summary ${name}/${key}`);
    assert(last.all_outside_correct === run.test.both_operands_and_answer_outside[0] &&
      last.all_outside_rows === run.test.both_operands_and_answer_outside[1],
    `Final outside-dictionary stratum differs ${name}/${key}`);
    const off = run.test.both_operands_and_answer_outside;
    rows.push([name, run.method ?? "rope", Boolean(summary.input_only), run.seed, run.dictionary_size, run.rotate, run.parameters, run.epochs,
      run.train.accuracy, run.test.accuracy, run.best_recorded_test_accuracy, ...off,
      run.first_recorded_test_99_epoch ?? "", run.elapsed_seconds].join(","));
    histories.push(...lines.map((line) => `${name},${run.seed},${run.dictionary_size},${run.rotate},${line}`));
  }
  series.push({ directory: name, ...summary });
}

const sources = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "src/simple/gpu.rs",
  "src/simple/mod.rs", "src/simple/anchor_modular.rs", "src/bin/anchor-div97.rs",
  "scripts/archive-anchor-div97.mjs", "target/release/anchor-div97"];
const inputOnly = series.every(s => s.input_only);
if (inputOnly) sources.push("src/bin/anchor-interpolation.rs", "target/release/anchor-interpolation");
const provenance = {
  date: "2026-10-16",
  git_head: gitHead(),
  source_state: "Uncommitted experiment changes; file hashes identify the executed implementation.",
  sha256: Object.fromEntries(sources.map((file) => [file, hash(file)])),
  rust_toolchain: fs.readFileSync("rust-toolchain.toml", "utf8").trim(),
  gpu: "NVIDIA GeForce RTX 3050 Laptop GPU, sm_86, 4096 MiB",
  validation: ["cargo test --lib --locked: 70 passed", "cargo clippy --all-targets --locked: passed",
    "cargo fmt --all -- --check: passed", inputOnly ? "anchor-interpolation --verify-only: passed" : "anchor-div97 --verify-only: passed",
    ...(inputOnly ? ["cargo test --bin anchor-interpolation --locked: 1 passed"] : [])],
  series,
};
fs.mkdirSync(destination, { recursive: true });
fs.writeFileSync(path.join(destination, "summary.json"), JSON.stringify(provenance, null, 2) + "\n");
fs.writeFileSync(path.join(destination, "runs.csv"),
  "series,method,input_only,seed,dictionary_size,rotate,parameters,epochs,train_accuracy,test_accuracy,best_recorded_test_accuracy,all_outside_correct,all_outside_rows,first_recorded_test_99_epoch,elapsed_seconds\n" + rows.join("\n") + "\n");
fs.writeFileSync(path.join(destination, "histories.csv.gz"), gzipSync(
  "series,seed,dictionary_size,rotate," + curveHeader + "\n" + histories.join("\n") + "\n"));
fs.copyFileSync(inputOnly ? "target/anchor-interpolation-verification.log" : "target/anchor-div97-verification.log",
  path.join(destination, "cuda-verification.log"));

for (const epochs of [...new Set(series.flatMap(s => s.runs.map(r => r.epochs)))]) {
  if (inputOnly) continue;
  console.log(`Epochs: ${epochs}`);
  console.log("| K | Parameters | Mean train | Mean test | Test range | All outside dictionary |");
  console.log("| ---: | ---: | ---: | ---: | --- | ---: |");
  const runs = series.flatMap(s => s.runs).filter(r => r.epochs === epochs && r.rotate);
  const percent = (value) => `${(100 * value).toFixed(2)}%`;
  for (const k of [...new Set(runs.map(r => r.dictionary_size))].sort((a, b) => b - a)) {
    const rs = runs.filter(r => r.dictionary_size === k);
    const mean = (key) => rs.reduce((sum, r) => sum + r[key].accuracy, 0) / rs.length;
    const accuracies = rs.map(r => r.test.accuracy);
    const off = rs.reduce((a, r) => a.map((x, i) => x + r.test.both_operands_and_answer_outside[i]), [0, 0]);
    console.log(`| ${k} | ${rs[0].parameters} | ${percent(mean("train"))} | ${percent(mean("test"))} | ${percent(Math.min(...accuracies))}–${percent(Math.max(...accuracies))} | ${off[1] ? percent(off[0] / off[1]) : "N/A"} |`);
  }
}
if (inputOnly) {
  console.log("Input-only interpolation: mean final held-out accuracy by method and K");
  for (const s of series) {
    const record = {};
    for (const k of [...new Set(s.runs.map(r => r.dictionary_size))].sort((a,b) => b-a)) {
      const rs = s.runs.filter(r => r.dictionary_size === k);
      record[k] = { train: rs.reduce((sum,r) => sum+r.train.accuracy,0)/rs.length,
        test: rs.reduce((sum,r) => sum+r.test.accuracy,0)/rs.length };
    }
    console.log(s.method, JSON.stringify(record));
  }
}
console.log(`Validated and archived ${rows.length} runs in ${destination}`);
