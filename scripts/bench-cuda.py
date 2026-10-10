#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run reproducible FP32 shapes and sample actual per-process GPU memory."""

import argparse
import csv
import statistics
import subprocess
import time
from pathlib import Path


SCENARIOS = {
    "baseline": [],
    "large-batch": ["--batch", "8192"],
    "wide": ["--width", "1024", "--batch", "1152"],
    "deep": ["--width", "256", "--layers", "20", "--batch", "512"],
    "mixed": ["--width", "512", "--layers", "4", "--batch", "1200"],
    "long-context": ["--context", "1024", "--batch", "16"],
    "stress-large-batch": ["--batch", "20000"],
    "stress-wide": ["--width", "1024", "--batch", "2688"],
    "stress-deep": ["--width", "256", "--layers", "49", "--batch", "512"],
    "stress-mixed": ["--width", "512", "--layers", "4", "--batch", "2816"],
    "stress-long-context": ["--context", "4096", "--batch", "6"],
}

DEFAULT_SCENARIOS = ["baseline", *(name for name in SCENARIOS if name.startswith("stress-"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/mesh-gpt-bench"))
    parser.add_argument("--output", type=Path, default=Path("target/mesh-gpt-profile/shapes"))
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w") as summary:
        writer = csv.writer(summary)
        writer.writerow(["scenario", "median_ms", "peak_process_mib", "allocated_device_bytes"])
        for name in args.scenario or DEFAULT_SCENARIOS:
            directory = args.output / name
            directory.mkdir(parents=True, exist_ok=True)
            command = [str(args.binary), *SCENARIOS[name], "--steps", str(args.steps),
                       "--warmup", str(args.warmup), "--trials", str(args.trials),
                       "--output", str(directory)]
            print(f"Running {name}: {' '.join(command)}", flush=True)
            probe = [str(args.binary), *SCENARIOS[name], "--steps", "20" if name == "baseline" else "2",
                     "--warmup", "1", "--trials", "1", "--output", str(directory / "memory-probe")]
            peak = 0
            with (directory / "memory-probe.log").open("w") as log, (directory / "memory.csv").open("w") as memory:
                samples = csv.writer(memory)
                samples.writerow(["elapsed_seconds", "process_mib"])
                start = time.monotonic()
                process = subprocess.Popen(probe, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        query = subprocess.run(
                            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                             "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True,
                        )
                        for row in csv.reader(query.stdout.splitlines()):
                            if row and row[0].strip() == str(process.pid):
                                used = int(row[1])
                                peak = max(peak, used)
                                samples.writerow([f"{time.monotonic() - start:.3f}", used])
                        time.sleep(0.1)
                finally:
                    if process.poll() is None:
                        process.terminate()
                    process.wait()
            if process.returncode:
                raise RuntimeError(f"{name} memory probe failed; see {directory / 'memory-probe.log'}")
            if not peak:
                raise RuntimeError(f"No GPU memory samples for {name}; see {directory / 'memory.csv'}")
            # Keep nvidia-smi sampling and its driver activity out of timed trials.
            with (directory / "run.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (directory / "timings.csv").open() as timings:
                median = statistics.median(float(row["ms_per_step"]) for row in csv.DictReader(timings))
            metadata = dict(line.split("=", 1) for line in (directory / "run.txt").read_text().splitlines())
            writer.writerow([name, f"{median:.6f}", peak, metadata["allocated_device_bytes"]])
            summary.flush()
            print(f"{name}: {median:.3f} ms/update, peak {peak} MiB", flush=True)


if __name__ == "__main__":
    main()
