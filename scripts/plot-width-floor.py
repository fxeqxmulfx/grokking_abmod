#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.10,<4", "numpy>=2,<3"]
# ///
"""Plot the finite below-W16 optimizer grid and boundary learning curves."""

import argparse
import csv
import gzip
import json
import math
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
                        default=Path("docs/grokking-results/dense-below16-grid"))
    directory = parser.parse_args().directory
    summary = json.loads((directory / "summary.json").read_text())
    plan, seeds = summary["plan"], summary["seeds"]
    results = list(reversed(summary["results"]))
    columns = min(2, len(results))
    figure, axes = plt.subplots(math.ceil(len(results) / columns), columns,
                                figsize=(5.5 * columns, 3.8 * math.ceil(len(results) / columns)),
                                squeeze=False, layout="constrained")
    norm = LogNorm(vmin=50, vmax=plan["epochs"])
    palette = plt.get_cmap("viridis").copy()
    palette.set_bad("#e8e9e3")
    for axis, result in zip(axes.flat, results):
        profiles = result["profiles"]
        rates = sorted({profile["learningRate"] for profile in profiles})
        decays = sorted({profile["weightDecay"] for profile in profiles})
        records = {(profile["learningRate"], profile["weightDecay"]): profile
                   for profile in profiles}
        values = np.full((len(rates), len(decays)), np.nan)
        for i, rate in enumerate(rates):
            for j, decay in enumerate(decays):
                profile = records.get((rate, decay))
                if profile is None:
                    label = "not searched"
                else:
                    run = next(run for run in profile["runs"] if run["seed"] == plan["screenSeed"])
                    if run["firstTest99"] is None:
                        label = f"peak {100 * run['peakTest']:.1f}%"
                    else:
                        values[i, j] = run["firstTest99"]
                        label = f"{run['firstTest99']} epochs\n{profile['successes']}/{len(seeds)} seeds"
                        if profile.get("completeSeeds", len(profile["runs"])) < len(seeds):
                            label += " *"
                color = "white" if np.isfinite(values[i, j]) and values[i, j] < 1800 else "black"
                axis.text(j, i, label, ha="center", va="center", fontsize=8.5, color=color)
        image = axis.imshow(values, norm=norm, cmap=palette, aspect="auto")
        axis.set_title(f"W{result['width']} · {result['parameters']:,} parameters")
        axis.set_xticks(range(len(decays)), [f"{value:g}" for value in decays])
        axis.set_yticks(range(len(rates)), [f"{value:g}" for value in rates])
        axis.set_xlabel("AdamW weight decay")
        axis.set_ylabel("Learning rate")
    for axis in list(axes.flat)[len(results):]:
        axis.set_visible(False)
    figure.colorbar(image, ax=axes, label=f"Epochs to first ≥99% test accuracy, seed {plan['screenSeed']}", shrink=.7)
    figure.suptitle(f"Rotate90 Dense width search · {plan['epochs']:,}-epoch budget\n"
                   "Color: seed 11 reached 99%; gray: peak test accuracy; *: confirmations stopped")
    save(figure, directory, "width-grid")

    minimum = summary["minimumRobustWidth"]
    widths = {16}
    if summary["minimumScreenWidth"] is not None:
        widths.add(summary["minimumScreenWidth"])
    if minimum is not None:
        widths.add(minimum)
        lower = [result["width"] for result in results if result["width"] < minimum]
        if lower:
            widths.add(max(lower))
    selected = [result for result in results if result["width"] in widths]
    figure, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4.5),
                                squeeze=False, layout="constrained")
    for axis, result in zip(axes.flat, selected):
        profile = result["winner"] or result["profiles"][0]
        for index, run in enumerate(profile["runs"]):
            with gzip.open(directory / run["directory"] / "loss.csv.gz", "rt") as stream:
                history = list(csv.DictReader(stream))
            epochs = [float(row["completed_epochs"]) for row in history]
            color = ["#49849c", "#dd6b3c", "#64916b"][index % 3]
            axis.plot(epochs, [100 * float(row["validation_accuracy"]) for row in history],
                      label=f"Test, seed {run['seed']}", color=color, linewidth=1.1)
            axis.plot(epochs, [100 * float(row["train_accuracy"]) for row in history],
                      color=color, linewidth=.7, linestyle="--", alpha=.5)
        axis.axhline(99, color="#555555", linewidth=.7, linestyle=":")
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlim(0, max(run["completedEpochs"] for run in profile["runs"]))
        axis.set_ylim(0, 101)
        axis.set_xlabel("Complete epochs (logarithmic after 10)")
        axis.set_ylabel("Accuracy (%)")
        axis.set_title(f"W{result['width']} · lr {profile['learningRate']}, wd {profile['weightDecay']}")
        axis.legend(loc="lower right", fontsize=8)
    figure.suptitle("W16 control and smaller-width attempts · dashed curves show training accuracy")
    save(figure, directory, "boundary-curves")


if __name__ == "__main__":
    main()
