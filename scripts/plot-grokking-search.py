#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.10,<4"]
# ///
"""Plot dense and Mesh GPT search histories against complete training epochs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["svg.hashsalt"] = "grokking-size-search"


def points(run, archive):
    if "history_file" in run:
        path = archive / run["history_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["plot_history_sha256"]:
            raise ValueError(f"Archived curve hash mismatch: {path}")
        with path.open(newline="") as source:
            return [(float(row["epoch"]), float(row["train"]), float(row["heldout"]))
                    for row in csv.DictReader(source)]
    directory = Path(run["run"])
    if run["family"] == "dense":
        rows = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
        return [(row["epochs_seen"], row["train"]["accuracy"], row["heldout"]["accuracy"]) for row in rows]
    settings = run["settings"]
    count, batch = int(settings["training_equations"]), int(settings["batch"])
    updates = (count + batch - 1) // batch
    with (directory / "loss.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    result = []
    for row in rows:
        complete, partial = divmod(int(row["step"]), updates)
        seen = int(row.get("training_examples_seen", complete * count + min(partial * batch, count)))
        result.append((seen / count, float(row["train_accuracy"]), float(row["validation_accuracy"])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = json.loads(arguments.summary.read_text())
    families = sorted({run["family"] for run in report["runs"]})
    figure, axes = plt.subplots(len(families), 2, figsize=(12, 4 * len(families)), squeeze=False,
                                layout="constrained")
    for index, family in enumerate(families):
        runs = sorted((run for run in report["runs"] if run["family"] == family),
                      key=lambda run: run["parameters"], reverse=True)
        peaks = [0.0, 0.0]
        for run in runs:
            history = points(run, arguments.summary.parent)
            epochs, train, heldout = zip(*history, strict=True)
            label = f"{Path(run['run']).name} ({run['parameters']:,} params)"
            if run["status"] == "interrupted":
                label += f" [stopped at {run['epochs']:g} epochs]"
            axes[index, 0].plot(epochs, [100 * value for value in train], label=label, linewidth=1)
            axes[index, 1].plot(epochs, [100 * value for value in heldout], label=label, linewidth=1)
            peaks[0] = max(peaks[0], max((value for epoch, value in zip(epochs, train, strict=True)
                                        if epoch >= 5000), default=0))
            peaks[1] = max(peaks[1], max((value for epoch, value in zip(epochs, heldout, strict=True)
                                        if epoch >= 5000), default=0))
        for axis, split, peak in zip(axes[index], ("Training", "Held-out"), peaks, strict=True):
            axis.axhline(99, color="gray", linestyle="--", linewidth=.6)
            axis.set(title=f"{family.title()} GPT: {split.lower()} accuracy",
                     xlabel="Complete passes over the training split", ylabel="Accuracy (%)",
                     xlim=(0, 10_000), ylim=(0, 101))
            axis.grid(alpha=.2)
            axis.legend(fontsize=7, loc="upper right" if peak < .65 else "lower right")
    figure.suptitle("Division modulo 97: exhaustive fixed-split evaluation")
    arguments.output.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        path = arguments.output / f"grokking-size-search.{extension}"
        metadata = {"Date": None} if extension == "svg" else {}
        figure.savefig(path, dpi=180, metadata=metadata)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
