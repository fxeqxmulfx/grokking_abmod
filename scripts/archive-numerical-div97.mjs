import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { gzipSync } from "node:zlib";

const [destination, ...directories] = process.argv.slice(2);
if (!destination || !directories.length) {
  throw new Error("Usage: node scripts/archive-numerical-div97.mjs OUTPUT RUN_DIRECTORY...");
}
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const hash = file => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const series = [];
const rows = [];
const histories = [];
const prefixes = new Map();
const codebooks = new Map();
let matchedPrefixRecords = 0;
let curveHeader;
for (const directory of directories) {
  const summary = JSON.parse(fs.readFileSync(path.join(directory, "summary.json"), "utf8"));
  assert(summary.status === "complete", `Incomplete series: ${directory}`);
  const label = path.basename(directory);
  const names = new Set();
  for (const run of summary.runs) {
    assert(!names.has(run.name), `Duplicate run ${label}/${run.name}`);
    names.add(run.name);
    const w = run.width;
    const expected = run.code_features*w + (run.encoder === "mlp" ? 3 : 2)*w*w + 97*w;
    assert(run.parameters === expected, `Parameter count: ${run.name}`);
    assert(run.whole_value_input_rows === (run.method === "table" ? 97 : 0), `Input table: ${run.name}`);
    for (const split of [run.train, run.test]) {
      assert(Number.isInteger(split.rows) && split.rows > 0 && Number.isInteger(split.correct) &&
        split.correct >= 0 && split.correct <= split.rows, `Split counts: ${run.name}`);
      assert(split.accuracy === split.correct/split.rows && Number.isFinite(split.cross_entropy), `Metrics: ${run.name}`);
      assert(split.zero_numerator[0] >= 0 && split.zero_numerator[0] <= split.zero_numerator[1] &&
        split.zero_numerator[1] <= split.rows, `Zero stratum: ${run.name}`);
      if (split.operand_strata) {
        assert(split.operand_strata.length === 4 &&
          split.operand_strata.every(([correct, rows]) => Number.isInteger(correct) && Number.isInteger(rows) && correct >= 0 && correct <= rows) &&
          split.operand_strata.reduce((sum, p) => sum+p[0], 0) === split.correct &&
          split.operand_strata.reduce((sum, p) => sum+p[1], 0) === split.rows,
        `Operand strata: ${run.name}`);
      }
    }
    if (run.training_operand_counts) {
      assert(run.training_operand_counts.length === 97 && run.training_answer_counts.length === 97 &&
        run.training_operand_counts.reduce((a,b) => a+b, 0) === 2*run.train.rows &&
        run.training_answer_counts.reduce((a,b) => a+b, 0) === run.train.rows,
      `Training exposure counts: ${run.name}`);
      if (run.heldout_start !== null) {
        assert(run.training_operand_counts.slice(run.heldout_start).every(n => n === 0) &&
          run.training_operand_counts.slice(0, run.heldout_start).every(n => n > 0) &&
          run.training_answer_counts.every(n => n > 0), `Held-out operand exposure: ${run.name}`);
        assert(run.train.operand_strata.slice(1).every(p => p[1] === 0), `Held-out operand leaked into training: ${run.name}`);
        if (summary.operation === "div") {
          const unseen = 97-run.heldout_start;
          assert(run.test.operand_strata[1][1] === unseen*(run.heldout_start-1) &&
            run.test.operand_strata[2][1] === run.heldout_start*unseen &&
            run.test.operand_strata[3][1] === unseen*unseen, `Held-out stratum coverage: ${run.name}`);
        }
      }
    }
    assert(run.train.rows + run.test.rows === (summary.operation === "div" ? 97*96 : 97*97), `Dataset size: ${run.name}`);
    const checkpointPath = path.join(directory, `${run.name}.f32`);
    const checkpoint = fs.readFileSync(checkpointPath);
    assert(checkpoint.length === run.parameters*4, `Checkpoint shape: ${run.name}`);
    for (let i = 0; i < checkpoint.length; i += 4) assert(Number.isFinite(checkpoint.readFloatLE(i)), `Nonfinite parameter: ${run.name}`);
    run.checkpoint_sha256 = hash(checkpointPath);
    if (run.method === "random-bits" || run.method === "integer-bits") {
      const filename = `${run.method}-h${run.harmonics}-codes.csv`;
      const codebookPath = path.join(directory, filename);
      const content = fs.readFileSync(codebookPath, "utf8");
      const codes = content.trim().split("\n");
      assert(codes.shift() === "value,bits" && codes.length === 97, `Binary codebook shape: ${run.name}`);
      const words = new Set();
      codes.forEach((line, value) => {
        const [index, bits] = line.split(",");
        const count = run.code_features - Number(run.method === "integer-bits");
        assert(Number(index) === value && bits.length === count && /^[01]+$/.test(bits),
          `Invalid binary code: ${run.name}/${value}`);
        if (run.method === "integer-bits") assert(parseInt(bits, 2) === value, `Incorrect positional digits: ${value}`);
        words.add(bits);
      });
      assert(words.size === 97, `Binary code collision: ${run.name}`);
      if (codebooks.has(filename)) {
        assert(codebooks.get(filename).content === content, `Different binary code assignments: ${filename}`);
      } else codebooks.set(filename, { source: codebookPath, content });
      run.codebook = `codebooks/${filename}`;
      run.codebook_sha256 = hash(codebookPath);
    }
    const lines = fs.readFileSync(path.join(directory, `${run.name}.csv`), "utf8").trim().split("\n");
    const header = lines.shift();
    curveHeader ??= header;
    assert(header === curveHeader, `Different curve columns: ${run.name}`);
    const columns = header.split(",");
    const records = lines.map(line => {
      const values = line.split(",").map(Number);
      assert(values.length === columns.length && values.every(Number.isFinite), `Curve values: ${run.name}`);
      return Object.fromEntries(columns.map((column, i) => [column, values[i]]));
    });
    assert(records[0].epoch === 0 && records.at(-1).epoch === run.epochs, `Incomplete curve: ${run.name}`);
    for (let i = 0; i < records.length; i++) {
      assert(records[i].updates === Math.ceil(run.train.rows/512)*records[i].epoch, `Update count: ${run.name}`);
      if (i) assert(records[i].epoch > records[i-1].epoch, `Curve order: ${run.name}`);
    }
    assert(Math.max(...records.map(r => r.test_accuracy)) === run.best_recorded_test_accuracy,
      `Best recorded accuracy: ${run.name}`);
    assert((records.find(r => r.test_accuracy >= 0.99)?.epoch ?? null) === run.first_recorded_test_99_epoch,
      `First recorded 99% epoch: ${run.name}`);
    const trajectoryKey = `${summary.operation}/${summary.split ?? "pairs"}/${summary.learning_rate}/${summary.weight_decay}/${run.name}`;
    const prefix = prefixes.get(trajectoryKey) ?? new Map();
    for (const record of records) {
      const { elapsed_seconds, ...deterministic } = record;
      const value = JSON.stringify(deterministic);
      if (prefix.has(record.epoch)) {
        assert(prefix.get(record.epoch) === value, `Different rerun trajectory at epoch ${record.epoch}: ${run.name}`);
        matchedPrefixRecords++;
      } else prefix.set(record.epoch, value);
    }
    prefixes.set(trajectoryKey, prefix);
    const last = records.at(-1);
    assert(last.train_accuracy === run.train.accuracy && last.test_accuracy === run.test.accuracy, `Final accuracy: ${run.name}`);
    assert(Math.abs(last.train_cross_entropy-run.train.cross_entropy) < 1e-12 &&
      Math.abs(last.test_cross_entropy-run.test.cross_entropy) < 1e-12, `Final loss: ${run.name}`);
    assert(last.zero_numerator_correct === run.test.zero_numerator[0] &&
      last.zero_numerator_rows === run.test.zero_numerator[1], `Final zero stratum: ${run.name}`);
    const strata = run.test.operand_strata ?? [[run.test.correct, run.test.rows], [0,0], [0,0], [0,0]];
    for (const [i, key] of ["familiar", "unseen_a", "unseen_b", "both_unseen"].entries()) {
      if (!(key+"_rows" in last)) continue;
      assert(last[key+"_correct"] === strata[i][0] && last[key+"_rows"] === strata[i][1], `Final operand stratum: ${run.name}/${key}`);
    }
    rows.push([label, summary.operation, summary.learning_rate, summary.weight_decay,
      run.method, run.encoder, run.width, run.harmonics, run.code_features, run.whole_value_input_rows,
      run.parameters, run.seed, run.epochs, run.updates, run.train.accuracy, run.test.accuracy,
      run.best_recorded_test_accuracy, run.first_recorded_test_99_epoch ?? "", run.elapsed_seconds,
      run.split ?? "pairs", run.heldout_start ?? "", ...strata.flat()].join(","));
    histories.push(...lines.map(line => `${label},${run.name},${line}`));
  }
  series.push({ directory: label, ...summary });
}
const sources = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "src/simple/mod.rs",
  "src/simple/gpu.rs", "src/simple/reference.rs", "src/simple/numerical.rs",
  "src/bin/numerical-div97.rs", "scripts/archive-numerical-div97.mjs", "target/release/numerical-div97"];
const head = fs.readFileSync(".git/HEAD", "utf8").trim();
let gitHead = head;
if (head.startsWith("ref: ")) {
  const ref = head.slice(5);
  const loose = path.join(".git", ref);
  gitHead = fs.existsSync(loose) ? fs.readFileSync(loose, "utf8").trim() :
    fs.readFileSync(".git/packed-refs", "utf8").split("\n").find(line => line.endsWith(` ${ref}`)).split(" ")[0];
}
const verificationLog = fs.readFileSync("target/numerical-div97-verification.log", "utf8");
const verified = new Set();
const verificationLines = verificationLog.trim().split("\n");
for (const line of verificationLines) {
  const match = /^PASS numerical (\S+) nonlinear=(false|true):/.exec(line);
  assert(match, "Incomplete CUDA verification log");
  const key = `${match[1]}/${match[2]}`;
  assert(!verified.has(key), `Duplicate CUDA verification: ${key}`);
  verified.add(key);
}
for (const s of series) for (const r of s.runs) {
  assert(verified.has(`${r.method}/false`) && verified.has(`${r.method}/true`),
    `Missing encoder verification: ${r.method}`);
}
const provenance = {
  date: "2026-10-16", git_head: gitHead,
  source_state: "Uncommitted experiment changes; hashes identify the executed implementation.",
  sha256: Object.fromEntries(sources.map(file => [file, hash(file)])),
  gpu: "NVIDIA GeForce RTX 3050 Laptop GPU, sm_86, 4096 MiB",
  validation: ["cargo test --lib --locked: 73 passed", "cargo test --bin numerical-div97 --locked: 2 passed", "cargo clippy --all-targets --locked: passed",
    "cargo fmt --all -- --check: passed", `numerical-div97 --verify-only: ${verificationLines.length} cases passed`],
  matched_prefix_records_across_budgets: matchedPrefixRecords,
  series,
};
fs.mkdirSync(destination, { recursive: true });
if (codebooks.size) {
  fs.mkdirSync(path.join(destination, "codebooks"), { recursive: true });
  for (const [filename, codebook] of codebooks) fs.copyFileSync(codebook.source, path.join(destination, "codebooks", filename));
}
fs.writeFileSync(path.join(destination, "summary.json"), JSON.stringify(provenance, null, 2)+"\n");
fs.writeFileSync(path.join(destination, "runs.csv"),
  "series,operation,learning_rate,weight_decay,method,encoder,width,harmonics,code_features,whole_value_input_rows,parameters,seed,epochs,updates,train_accuracy,test_accuracy,best_recorded_test_accuracy,first_recorded_test_99_epoch,elapsed_seconds,split,heldout_start,familiar_correct,familiar_rows,unseen_a_correct,unseen_a_rows,unseen_b_correct,unseen_b_rows,both_unseen_correct,both_unseen_rows\n"+rows.join("\n")+"\n");
fs.writeFileSync(path.join(destination, "histories.csv.gz"), gzipSync("series,name,"+curveHeader+"\n"+histories.join("\n")+"\n"));
fs.copyFileSync("target/numerical-div97-verification.log", path.join(destination, "cuda-verification.log"));
for (const s of series) {
  console.log(s.directory, s.operation, `lr=${s.learning_rate} wd=${s.weight_decay}`);
  const groups = Map.groupBy(s.runs, r => `${r.method} ${r.encoder} W${r.width} H${r.harmonics} epochs=${r.epochs}`);
  for (const [name, runs] of groups) {
    const mean = split => runs.reduce((a,r) => a+r[split].accuracy, 0)/runs.length;
    console.log(`${name} F=${runs[0].code_features} parameters=${runs[0].parameters}: train=${(100*mean("train")).toFixed(2)}% test=${(100*mean("test")).toFixed(2)}% n=${runs.length}`);
  }
}
console.log(`Validated and archived ${rows.length} runs in ${destination}`);
