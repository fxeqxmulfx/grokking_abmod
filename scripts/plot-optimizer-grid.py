#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.10,<4", "numpy>=2,<3"]
# ///
"""Plot the finite Rust/CUDA optimizer grid and independent-seed confirmations."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def save(figure, directory, name):
    for extension in ("svg", "png"):
        figure.savefig(directory / f"{name}.{extension}", dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path("docs/grokking-results/dense-optimizer-grid"))
    args = parser.parse_args()
    directory = args.directory
    summary = json.loads((directory / "summary.json").read_text())
    runs = json.loads((directory / "runs.json").read_text())
    plan = summary["plan"]
    rates, decays = plan["learningRates"], plan["weightDecays"]

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for axis, width in zip(axes.flat, plan["widths"]):
        records = {(run["learningRate"], run["weightDecay"]): run
                   for run in runs if run["stage"] == "screen" and run["width"] == width}
        values = np.array([[records[rate, decay]["firstTest99"] or np.nan
                            for decay in decays] for rate in rates])
        image = axis.imshow(values, norm=LogNorm(vmin=50, vmax=2000), cmap="viridis")
        axis.set_title(f"W{width} · seed {plan['screenSeed']}")
        axis.set_xticks(range(len(decays)), [str(value) for value in decays])
        axis.set_yticks(range(len(rates)), [str(value) for value in rates])
        axis.set_xlabel("AdamW weight decay")
        axis.set_ylabel("Learning rate")
        for i, rate in enumerate(rates):
            for j, decay in enumerate(decays):
                run = records[rate, decay]
                label = str(run["firstTest99"]) if run["firstTest99"] is not None else f">{run['budget']}\npeak {100*run['peakTest']:.1f}%"
                axis.text(j, i, label, ha="center", va="center", fontsize=9,
                          color="black" if np.isnan(values[i, j]) or values[i, j] > 500 else "white")
    figure.colorbar(image, ax=axes, label="Complete epochs to first ≥99% held-out accuracy")
    figure.suptitle("Rust/CUDA · finite optimizer grid (censored trials show peak accuracy)")
    save(figure, directory, "screen-grid")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for axis, result in zip(axes.flat, summary["results"]):
        selected = result["winner"] or result["profiles"][0]
        baseline = next(profile for profile in result["profiles"]
                        if all(profile[key] == plan["baseline"][key] for key in ("learningRate", "weightDecay")))
        for profile, color, label in [(baseline, "#888888", "Original"), (selected, "#49849c", "Selected")]:
            for i, run in enumerate(profile["runs"]):
                with gzip.open(directory / run["directory"] / "loss.csv.gz", "rt") as stream:
                    rows = list(csv.DictReader(stream))
                axis.plot([float(row["completed_epochs"]) for row in rows],
                          [100*float(row["validation_accuracy"]) for row in rows],
                          color=color, alpha=0.7, linewidth=1.2,
                          label=label if i == 0 else None)
        axis.set_title(f"W{result['width']} · lr={selected['learningRate']}, wd={selected['weightDecay']}")
        axis.set_xscale("symlog", linthresh=10)
        axis.set_ylim(0, 101)
        axis.axhline(99, color="#dd6b3c", linewidth=0.7, linestyle="--")
        axis.set_xlabel("Complete epochs (logarithmic after 10)")
        axis.set_ylabel("Held-out accuracy (%)")
        axis.legend(loc="lower right")
    figure.suptitle("Original versus selected profiles · seeds 11, 17 and 23")
    save(figure, directory, "confirmed-curves")


if __name__ == "__main__":
    main()
