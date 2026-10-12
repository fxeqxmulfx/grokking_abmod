# Continuous search for delayed, sharp mesh grokking

`sharp-mesh-search` is a host-only Rust scheduler. It starts one Rust/CUDA
`mesh-gpt` process at a time and searches only mesh architectures: ordinary
mesh MLPs, MLPs with a shared operand/readout embedding, residual mesh MLPs,
and Transformers with mesh attention/feed-forward projections. Table embeddings
are allowed for the Transformer; its channel mixers still use mesh edges.

The current short-plateau study uses a 25-epoch minimum and evaluates every
epoch, with prioritized replications across seeds and compact variants. See
[measured short-plateau results](short-mesh-results.md). The original
1,000-epoch-minimum study is preserved separately.

## Acceptance rule

The defaults require all of the following, in this order:

1. A contiguous interval of **at least 1,000 epochs** with train accuracy
   **at least 99%** and held-out accuracy **at most 10%**.
2. The first held-out crossing of **90%**, occurring **at most 250 epochs**
   after the last report at or below 10% before that crossing.
3. A subsequent held-out accuracy of **at least 99%**.

The scheduler evaluates the saved curve rather than treating a late 99%
crossing as proof of a sharp transition. Train accuracy dips break the plateau.
Missing reports break it too. Later regressions do not overwrite the first
90% crossing. A model that generalizes immediately cannot qualify.

Both complete splits are evaluated every 250 updates, or 25 epochs in this
protocol. Plateau and crossing times are **sampled observations**, with this
25-epoch resolution. A qualifying result is an observed candidate on the
selection split, not yet a replication across seeds.

The task is division modulo 97, with 4,656 training and 4,656 held-out
equations. Each epoch visits every training equation once: nine batches of
512 and a final batch of 48. Each trial has a fresh model, sampler, and AdamW
state. The cap is 10,000 epochs, with an earlier stop only at 99% held-out
accuracy, invalid/nonfinite metrics, a process error, a requested stop, or a
one-hour wall-time limit. Timeouts and interrupted runs are not full-budget
negative results. There is no 50-epoch underfitting cutoff.

## Build and start

Run these commands from the repository root, with access to the NVIDIA driver:

```bash
# Compile the trainer with embedded CUDA kernels first.
cargo oxide build --arch sm_86 -- --release --locked --bin mesh-gpt

# This scheduler needs only the normal host compiler.
cargo build --release --locked --bin sharp-mesh-search
target/release/sharp-mesh-search --dry-run 12
target/release/sharp-mesh-search
```

The loop runs until stopped. To stop automatically after the first qualifying
model, add `--stop-after-matches 1`. `--max-trials N` limits new trials in one
invocation. `--plateau-epochs`, `--jump-epochs`, `--epochs`, and `--search-seed`
are configurable. Use a new `--output` directory when changing the policy.

`--report-every N` must be a positive multiple of ten updates, so reports fall
on complete epochs. `--queue PATH` supplies prioritized argument lines before
the usual proposals. The queue is hashed into the policy and copied into the
search directory. Its lines are parsed as argument arrays without shell
evaluation; output and budget overrides and assisted operand encodings are
rejected.

To reproduce the active short-plateau search:

```bash
target/release/sharp-mesh-search \
  --output target/grok-div97-search/short-plateau-mesh-loop \
  --trainer target/grok-div97-search/sharp-mesh-loop/bin/mesh-gpt \
  --plateau-epochs 25 --report-every 10 \
  --queue experiments/div97-short-plateau-mesh.args
```

The prioritized queue contains 22 trials: replications of the two observed
candidates, Square residual meshes at widths 64 and 96, and smaller ReLU
variants. After the queue finishes, the continuous exploration resumes.

The initial proposals interleave Square residual meshes, deeper ReLU residual
meshes, tied own-channel meshes, and mesh Transformers with different initial
embedding scales and weight decay. Later proposals sample widths, depths,
connectivity, learning rates, decay, seeds, and Transformer settings using
SplitMix64. One proposal in four mutates the best observed configuration when
one is available. Completed delayed transitions rank ahead of underfit
plateaus, and qualifying sharp transitions rank first. One-hot operands and
token-ordered parity embeddings do not supply discrete logarithms as inputs.

## Inspect, stop, and resume

The default directory is `target/grok-div97-search/sharp-mesh-loop/`:

The active short-plateau study uses
`target/grok-div97-search/short-plateau-mesh-loop/` instead, with the same file
layout. Stop it with
`touch target/grok-div97-search/short-plateau-mesh-loop/STOP`.

| Path | Contents |
| --- | --- |
| `state.json` | Live trial, trainer PID, accuracy, plateau and crossing statistics |
| `results.jsonl` | Every finished, failed, diverged, timed-out, or interrupted attempt |
| `matches.jsonl` | Completed trials that satisfy all acceptance conditions; created on the first match |
| `best.json` | Most promising completed curve so far; this can be a nonqualifying result |
| `trials/<label>/loss.csv` | Full train/held-out accuracy and loss history |
| `trials/<label>/model.bin` | Latest weights; Adam moments are not included |
| `trials/<label>/arguments.json` | Exact argument array, without shell interpretation |
| `trials/<label>/run.txt` | Trainer metadata, parameter count, and stop reason |
| `trials/<label>/console.log` | Trainer output and errors |
| `trials/<label>/outcome.json` | Scheduler outcome and measured transition statistics |
| `policy.txt`, `bin/`, `sources/` | Fixed criteria, isolated binaries, hashes, and source snapshot |
| `sessions/` | Actual host scheduler executable, source, and hash for each invocation |

```bash
cat target/grok-div97-search/sharp-mesh-loop/state.json
tail -n 5 target/grok-div97-search/sharp-mesh-loop/results.jsonl

# Stop the active trial and prevent further launches.
touch target/grok-div97-search/sharp-mesh-loop/STOP

# Resume the search after it has stopped.
rm target/grok-div97-search/sharp-mesh-loop/STOP
target/release/sharp-mesh-search
```

Restarting skips completed configurations and retries interrupted proposals
from initialization, preserving earlier attempts in separate directories.
It does not resume training from a weights-only checkpoint. An operating-system
file lock prevents two workers using the same directory and releases even
after a process crash. Source and CUDA trainer copies isolate an active search
from later builds. Three consecutive trainer process failures stop the loop
so that an unavailable driver does not produce an endless error stream.

## Verification

```bash
cargo test --locked --bin sharp-mesh-search
cargo clippy --locked --bin sharp-mesh-search -- -D warnings
```

The tests cover classification of sharp, gradual, and immediate generalization;
train dips; partial CSV writes; missing observations; nonfinite metrics;
configuration validity and reproducibility; directory locking; stopping the
owned child process; and preservation of interrupted attempts on restart.

The new residual mesh and tied concatenation CUDA kernels were separately
checked against the independent f64 reference with `grok-simple --verify`.
Existing measured mesh studies are archived under
[`grokking-results/residual-mesh/`](grokking-results/residual-mesh/),
[`grokking-results/mesh-mlp-width/`](grokking-results/mesh-mlp-width/), and
[`grokking-results/table-unit-control/`](grokking-results/table-unit-control/).
Their curves motivate the search but are not reported as satisfying the
stricter acceptance rule above.
