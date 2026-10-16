#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.10,<4", "numpy>=2,<3"]
# ///
"""Plot local Rust/CUDA residual and scalar-regression experiments."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

COLORS = ["#347f9c", "#dc6b3d", "#518867"]


def save(figure, directory, name):
    for extension in ("png", "svg"):
        figure.savefig(directory / f"{name}.{extension}", dpi=170)
    plt.close(figure)


def history(run):
    with (Path(run["directory"]) / "loss.csv").open() as stream:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def plot_rotary(directory):
    path = directory / "rotary/summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for axis, profile in zip(axes, summary["plan"]["profiles"]):
        own = [run for run in summary["runs"] if run["profile"] == profile["name"]]
        for run in own:
            rows = history(run)
            index = summary["seeds"].index(run["seed"])
            axis.plot([row["completed_epochs"] for row in rows],
                      [100 * row["validation_accuracy"] for row in rows],
                      color=COLORS[index], linestyle="-" if run["residual"] else "--",
                      label=f"Seed {run['seed']}" if run["residual"] else None,
                      linewidth=1.1)
        axis.axhline(99, color="#777777", linestyle=":", linewidth=.7)
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlim(0, max(run["completedEpochs"] for run in own))
        axis.set_ylim(0, 101)
        axis.set_title(f"{profile['name'].title()} · lr {profile['learningRate']}, wd {profile['weightDecay']}")
        axis.set_xlabel("Complete epochs (logarithmic after 10)")
        axis.set_ylabel("Held-out accuracy (%)")
        axis.legend(fontsize=8)
    figure.suptitle("W128 skip ablation · solid: residual; dashed: no residual\n"
                   "Paired initialization, split and minibatch order; 10,000-epoch cap")
    save(figure, directory, "rotary-residual")


def config_key(run):
    fields = ["operation", "scale", "width", "depth", "preRmsnorm", "residual",
              "learningRate", "weightDecay"]
    return tuple(run[field] for field in fields)


def plot_scalar(directory):
    path = directory / "scalar/summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    plan = summary["plan"]["scalarMlp"]
    tasks = [(operation, scale) for operation in plan["operations"] for scale in plan["scales"]]
    all_runs = json.loads((directory / "scalar/runs.json").read_text())
    flags = [(norm, skip) for norm in (False, True) for skip in (False, True)]
    names = ["Plain", "Skip", "Pre-RMSNorm", "Pre-RMSNorm + skip"]

    for metric, ylabel, output_name, percentage in [
        ("Rmse", "RMSE in residue units", "scalar-rmse", False),
        ("Accuracy", "Rounded answer accuracy (%)", "scalar-accuracy", True),
    ]:
        figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
        for axis, (operation, scale) in zip(axes.flat, tasks):
            own = [result for result in summary["results"] if result["family"]["operation"] == operation and result["family"]["scale"] == scale]
            labels = []
            for x, result in enumerate(own):
                screen = result["screen"]
                labels.append(f"{names[x]}\nW{screen['width']} D{screen['depth']}")
                for index, run in enumerate(result["runs"]):
                    if run["status"] != "complete":
                        axis.text(x, 0, "diverged", ha="center", va="bottom", rotation=90)
                        continue
                    offset = (index - 1) * .12
                    factor = 100 if percentage else 1
                    axis.scatter(x + offset, factor * run[f"finalTest{metric}"], color=COLORS[index], s=38,
                                 label=f"Held-out, seed {run['seed']}" if x == 0 else None)
                    axis.scatter(x + offset, factor * run[f"finalTrain{metric}"], edgecolor=COLORS[index],
                                 facecolor="none", s=30, alpha=.7)
            baseline = np.mean([run[f"constantTest{metric}"] for result in own for run in result["runs"]])
            axis.axhline((100 if percentage else 1) * baseline, color="#777777", linestyle=":", linewidth=1,
                        label="Train-mean constant baseline")
            axis.set_xticks(range(len(own)), labels, fontsize=8)
            axis.set_title(f"{operation} mod 97 · {scale} inputs and target")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=.2)
            axis.set_ylim(bottom=0)
            axis.legend(fontsize=7)
        figure.suptitle(f"Scalar MLP · {plan['finalEpochs']:,} complete epochs · filled: held-out; hollow: training\n"
                       "Configuration selected on seed 11 in each family; D1 is the shared baseline")
        save(figure, directory, output_name)

    figure, axes = plt.subplots(4, 4, figsize=(16, 13), layout="constrained")
    for row, (operation, scale) in enumerate(tasks):
        for column, (norm, skip) in enumerate(flags):
            axis = axes[row, column]
            values = np.full((len(plan["depths"]), len(plan["widths"])), np.nan)
            for y, depth in enumerate(plan["depths"]):
                for x, width in enumerate(plan["widths"]):
                    candidates = [run for run in all_runs if run["stage"] == "screen" and run["status"] == "complete"
                                  and run["operation"] == operation and run["scale"] == scale
                                  and run["width"] == width and run["depth"] == depth
                                  and (depth == 1 or (run["preRmsnorm"] == norm and run["residual"] == skip))]
                    if candidates:
                        best = min(candidates, key=lambda run: run["finalTestRmse"])
                        values[y, x] = best["finalTestRmse"] / best["constantTestRmse"]
                        text = f"{best['finalTestRmse']:.2f}\n{100 * best['finalTestAccuracy']:.1f}%"
                    else:
                        text = "diverged"
                    axis.text(x, y, text, ha="center", va="center", fontsize=8)
            image = axis.imshow(values, cmap="RdYlGn_r", norm=LogNorm(vmin=.5, vmax=2), aspect="auto")
            axis.set_xticks(range(len(plan["widths"])), plan["widths"])
            axis.set_yticks(range(len(plan["depths"])), plan["depths"])
            axis.set_xlabel("Width")
            axis.set_ylabel("Hidden depth")
            axis.set_title(f"{operation}/{scale} · {names[column]}", fontsize=10)
    figure.colorbar(image, ax=axes, label="Held-out RMSE / train-mean baseline RMSE", shrink=.65)
    figure.suptitle(f"{summary['screenConfigurations']} scalar screens · {plan['screenEpochs']:,} epochs, seed 11\n"
                   "Each cell picks lr/wd with lowest final held-out RMSE; labels: residue RMSE and rounded accuracy")
    save(figure, directory, "scalar-grid")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, (operation, scale) in zip(axes.flat, tasks):
        own = [result for result in summary["results"] if result["family"]["operation"] == operation and result["family"]["scale"] == scale
               and all(run["status"] == "complete" for run in result["runs"])]
        result = min(own, key=lambda result: np.mean([run["finalTestRmse"] for run in result["runs"]]))
        config = result["screen"]
        for index, run in enumerate(result["runs"]):
            rows = history(run)
            axis.plot([row["epoch"] for row in rows], [row["test_rmse_residue"] for row in rows],
                      color=COLORS[index], label=f"Seed {run['seed']}", linewidth=1)
            axis.plot([row["epoch"] for row in rows], [row["train_rmse_residue"] for row in rows],
                      color=COLORS[index], linestyle="--", alpha=.6, linewidth=.8)
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlim(0, plan["finalEpochs"])
        axis.set_yscale("log")
        axis.set_xlabel("Complete epochs (logarithmic after 10)")
        axis.set_ylabel("RMSE in residue units (log scale)")
        axis.set_title(f"{operation}/{scale} · W{config['width']} D{config['depth']}\n"
                       f"norm={config['preRmsnorm']}, skip={config['residual']}, lr={config['learningRate']}, wd={config['weightDecay']}", fontsize=10)
        axis.legend(fontsize=8)
    figure.suptitle("Best mean held-out RMSE among the selected families · dashed: training")
    save(figure, directory, "scalar-learning-curves")

    figure, axes = plt.subplots(4, 3, figsize=(13, 15), layout="constrained")
    residue_cmap = plt.get_cmap("viridis").copy()
    residue_cmap.set_bad("#ececec")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("#ececec")
    for row, (operation, scale) in enumerate(tasks):
        own = [result for result in summary["results"]
               if result["family"]["operation"] == operation and result["family"]["scale"] == scale
               and all(run["status"] == "complete" for run in result["runs"])]
        result = min(own, key=lambda item: np.mean([run["finalTestRmse"] for run in item["runs"]]))
        run = next(run for run in result["runs"] if run["seed"] == plan["screenSeed"])
        target, predicted = np.full((97, 97), np.nan), np.full((97, 97), np.nan)
        with (Path(run["directory"]) / "predictions.csv").open() as stream:
            for point in csv.DictReader(stream):
                a, b = int(point["a"]), int(point["b"])
                target[b, a] = float(point["target"])
                predicted[b, a] = float(point["prediction_residue"])
        for column, (values, title) in enumerate([
            (target, "Correct residue"), (predicted, "MLP prediction"),
            (np.abs(predicted - target), "Absolute error"),
        ]):
            axis = axes[row, column]
            artist = axis.imshow(values, origin="lower", vmin=0, vmax=96,
                                 cmap=error_cmap if column == 2 else residue_cmap,
                                 interpolation="nearest")
            axis.set_title(f"{operation}/{scale} · {title}")
            axis.set_xlabel("Operand a")
            axis.set_ylabel("Operand b")
            figure.colorbar(artist, ax=axis, shrink=.72, extend="max" if column == 2 else "both")
        axes[row, 1].set_xlabel(f"Operand a · train {100 * run['finalTrainAccuracy']:.2f}% / held-out {100 * run['finalTestAccuracy']:.2f}%")
    figure.suptitle("Best mean-RMSE selected family per task · seed 11, held-out equations only\n"
                   "Gray cells are absent from this held-out split; colors use original residue units")
    save(figure, directory, "scalar-predictions")


def plot_investigation(directory):
    path = directory / "mlp-investigation.json"
    if not path.exists():
        path = directory / "mlp-investigation-partial.json"
    if not path.exists():
        return
    report = json.loads(path.read_text())
    pending = report["status"] != "complete"
    prefix = "mlp-partial" if pending else "mlp"
    label = "IN PROGRESS · " if pending else ""
    runs = report["runs"]

    for metric, ylabel, output_name, factor in [
        ("Accuracy", "Rounded answer accuracy (%)", "best-accuracy", 100),
        ("Rmse", "RMSE in residue units", "best-rmse", 1),
    ]:
        figure, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
        for axis, task in zip(axes.flat, report["bestByTask"]):
            group = task.get("trainingAccuracyFit", task["trainingFit"]) if metric == "Accuracy" else task["trainingFit"]
            if group is None:
                axis.text(.5, .5, f"{task['operation']}/{task['scale']}\nNo three-seed confirmation",
                          ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
                continue
            config = group["configuration"]
            for index, run in enumerate(group["runs"]):
                axis.scatter(index - .1, factor * run[f"finalTrain{metric}"], color=COLORS[0], s=45,
                             label="Training" if index == 0 else None)
                axis.scatter(index + .1, factor * run[f"finalTest{metric}"], edgecolor=COLORS[1], facecolor="none", s=45,
                             label="Held-out" if index == 0 else None)
            axis.set_xticks(range(3), [f"Seed {run['seed']}" for run in group["runs"]])
            axis.set_xlim(-.5, 2.5)
            values = sorted(factor * run[f"final{split}{metric}"]
                            for run in group["runs"] for split in ("Train", "Test"))
            if metric == "Accuracy":
                axis.axhline(95, color="#777777", linestyle=":", linewidth=.8)
                upper = 101
            else:
                upper = max(1, 1.1 * values[-1])
            axis.set_ylim(0, upper)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=.2)
            boundaries = [0, *values, upper]
            low, high = max(zip(boundaries, boundaries[1:]), key=lambda pair: pair[1] - pair[0])
            axis.legend(fontsize=8, loc="center right",
                        bbox_to_anchor=(.98, (low + high) / (2 * upper)))
            axis.set_title(f"{task['operation']}/{task['scale']} · W{config['width']} D{config['depth']} · bias {config['biasInitialization']}\n"
                           f"norm={config['preRmsnorm']}, skip={config['residual']}, lr={config['learningRate']}, wd={config['weightDecay']}\n"
                           f"{config['schedule']}, beta2={config['beta2']}, batch={config['batch']}; {group['epochs']:,} epochs", fontsize=8)
        selected_metric = "accuracy" if metric == "Accuracy" else "RMSE"
        figure.suptitle(label + f"Best confirmed training {selected_metric} per task across all completed phases\n"
                       "Fixed configurations, seeds 11/17/23; original mod-97 training halves"
                       + ("; dotted line: 95%" if metric == "Accuracy" else ""))
        save(figure, directory, prefix + "-" + output_name)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")
    for column, operation in enumerate(["div", "mul"]):
        candidates = [group for group in report["threeSeed"]
                      if group["complete"] and group["configuration"]["operation"] == operation]
        if not candidates:
            for axis in axes[:, column]:
                axis.set_axis_off()
            continue
        group = max(candidates, key=lambda item: item["meanTrainAccuracy"])
        config = group["configuration"]
        for color, run in zip(COLORS, group["runs"]):
            rows = history(run)
            epochs = [point["epoch"] for point in rows]
            for row, (metric, factor) in enumerate([("accuracy", 100), ("rmse_residue", 1)]):
                axis = axes[row, column]
                axis.plot(epochs, [factor * point[f"train_{metric}"] for point in rows],
                          color=color, label=f"Seed {run['seed']}")
                axis.plot(epochs, [factor * point[f"test_{metric}"] for point in rows],
                          color=color, linestyle="--")
        axes[0, column].set_title(
            f"{operation}/{config['scale']} · W{config['width']} D{config['depth']} · bias {config['biasInitialization']}\n"
            f"norm={config['preRmsnorm']}, skip={config['residual']}, lr={config['learningRate']}, wd={config['weightDecay']}\n"
            f"{config['schedule']}, beta2={config['beta2']}, batch={config['batch']}", fontsize=9)
        axes[0, column].axhline(95, color="#777777", linestyle=":", linewidth=.8)
        axes[0, column].set_ylim(0, 101)
        axes[0, column].set_ylabel("Rounded answer accuracy (%)")
        axes[0, column].legend(fontsize=8, loc="upper left")
        axes[1, column].set_yscale("log")
        axes[1, column].set_ylabel("RMSE in residue units (log scale)")
        for axis in axes[:, column]:
            axis.set_xlim(0, group["epochs"])
            axis.set_xlabel("Complete training epochs")
            axis.grid(alpha=.2)
    figure.suptitle(label + "Best confirmed training-fit recipes · solid: training, dashed: held-out; dotted: 95%\n"
                   "Fixed configurations, seeds 11/17/23; selected by mean final rounded training accuracy")
    save(figure, directory, prefix + "-learning-curves")

    controls = report["bestTrainingControls"]
    if controls:
        datasets = [(7, None), (31, None), (97, 64), (97, 256), (97, None)]
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for axis, operation in zip(axes, ["div", "mul"]):
            for color, scale in zip(COLORS, ["unit", "centered"]):
                own = {(run["modulus"], run["trainLimit"]): run for run in controls
                       if run["operation"] == operation and run["scale"] == scale}
                for metric, style, face in [("finalTrainAccuracy", "-", color), ("finalTestAccuracy", "--", "white")]:
                    values = [100 * own[dataset][metric] if dataset in own else np.nan for dataset in datasets]
                    axis.plot(range(5), values, marker="o", linestyle=style, color=color, markerfacecolor=face,
                              label=scale + (" training" if metric == "finalTrainAccuracy" else " held-out"))
            axis.set_xticks(range(5), ["mod 7\nfull half", "mod 31\nfull half", "mod 97\n64 rows", "mod 97\n256 rows", "mod 97\nfull half"])
            axis.set_ylim(0, 101)
            axis.set_title(operation)
            axis.set_ylabel("Rounded answer accuracy (%)")
            axis.grid(axis="y", alpha=.2)
            axis.legend(fontsize=8)
        figure.suptitle(label + "Capacity/task controls · W128 D4 · best final training RMSE per dataset and scale\n"
                       "10,000 updates each, seed 11; epochs and visits per equation differ with dataset size")
        save(figure, directory, prefix + "-controls")

    bias_runs = [run for run in runs if run["phase"] == "mlp-bias-screen"]
    if bias_runs:
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        tasks = [(operation, scale) for operation in ["div", "mul"] for scale in ["unit", "raw", "centered"]]
        modes = ["zero", "uniform", "domain"]
        for axis, (operation, scale) in zip(axes.flat, tasks):
            for color, lr in zip(COLORS, [.001, .01]):
                own = {run["biasInitialization"]: run for run in bias_runs if run["operation"] == operation
                       and run["scale"] == scale and run["learningRate"] == lr and run["status"] == "complete"}
                for metric, style in [("finalTrainRmse", "-"), ("finalTestRmse", "--")]:
                    values = [own[mode][metric] if mode in own else np.nan for mode in modes]
                    axis.plot(range(3), values, marker="o", linestyle=style, color=color,
                              markerfacecolor=color if metric == "finalTrainRmse" else "white",
                              label=f"lr {lr} " + ("training" if metric == "finalTrainRmse" else "held-out"))
            axis.set_xticks(range(3), modes)
            axis.set_title(f"{operation}/{scale}")
            axis.set_ylabel("RMSE in residue units")
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", alpha=.2)
            axis.legend(fontsize=7)
        figure.suptitle(label + "Paired initialization · W128 D4, no norm/skip, 1,000 epochs, seed 11\n"
                       "Identical Kaiming matrices; uniform/domain also share every later-layer bias")
        save(figure, directory, prefix + "-bias")

    ablations = report.get("optimizationAblations", [])
    if ablations:
        variants = ["depth4", "zero-bias", "uniform-bias", "beta2-098", "constant-lr", "batch512", "width256"]
        labels = ["Depth\n4", "Bias\nzero", "Bias\nuniform", "Beta2\n0.98", "LR\nconstant", "Batch\n512", "Width\n256"]
        figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained")
        for axis, operation in zip(axes, ["div", "mul"]):
            own = {run["variant"]: run for run in ablations if run["operation"] == operation}
            for offset, field, color, face, name in [
                (-.08, "trainAccuracy", COLORS[0], COLORS[0], "Training"),
                (.08, "testAccuracy", COLORS[1], "none", "Held-out"),
            ]:
                values = [100 * own[variant][field] if variant in own and own[variant]["status"] == "complete" else np.nan for variant in variants]
                axis.scatter(np.arange(len(variants)) + offset, values, edgecolor=color, facecolor=face, s=40, label=name)
            reference = next(run for run in runs if run["phase"] == "mlp-fullbatch-pilot" and run["operation"] == operation)
            axis.axhline(100 * reference["finalTrainAccuracy"], color="#666666", linestyle="--", linewidth=1, label="Reference train")
            axis.axhline(95, color="#999999", linestyle=":", linewidth=.8)
            for index, variant in enumerate(variants):
                if variant not in own or own[variant]["status"] != "complete":
                    text = "diverged" if variant in own else "pending"
                    axis.text(index, 6, text, rotation=90, ha="center", fontsize=7, color="#777777")
            axis.set_xticks(range(len(labels)), labels)
            axis.set_ylim(0, 101)
            axis.set_title(operation)
            axis.set_ylabel("Rounded answer accuracy (%)")
            axis.grid(axis="y", alpha=.2)
            axis.legend(fontsize=8)
        figure.suptitle(label + "One-factor comparisons · 10,000 complete epochs, seed 11\n"
                       "Reference: W128 D8, centered, domain bias, no norm/skip, full batch, lr 0.01 + cosine, beta2 0.999, wd 0\n"
                       "Batch 512 uses 100,000 updates; all other cases use 10,000. Dotted line: 95%")
        save(figure, directory, prefix + "-optimization-factors")

    deep = [run for run in runs if run["phase"] == "mlp-deep" and run["stage"] == "screen"]
    if deep:
        plan = json.loads((directory / "mlp-deep/plan.json").read_text())["deep"]
        tasks = [(operation, scale) for operation in plan["operations"] for scale in plan["scales"]]
        flags = [(norm, skip) for norm in (False, True) for skip in (False, True)]
        titles = ["Plain", "Skip", "Pre-RMSNorm", "Pre-RMSNorm + skip"]
        figure, axes = plt.subplots(4, 4, figsize=(15, 13), layout="constrained")
        for row, (operation, scale) in enumerate(tasks):
            for column, (norm, skip) in enumerate(flags):
                axis = axes[row, column]
                values = np.full((len(plan["depths"]), len(plan["widths"])), np.nan)
                for y, depth in enumerate(plan["depths"]):
                    for x, width in enumerate(plan["widths"]):
                        own = [run for run in deep if run["operation"] == operation and run["scale"] == scale
                               and run["depth"] == depth and run["width"] == width
                               and run["preRmsnorm"] == norm and run["residual"] == skip]
                        finite = [run for run in own if run["status"] == "complete"]
                        text = "pending" if len(own) < len(plan["learningRates"]) * len(plan["weightDecays"]) else "diverged"
                        if finite:
                            best = min(finite, key=lambda run: run["finalTrainRmse"])
                            values[y, x] = best["finalTrainRmse"]
                            text = f"{best['finalTrainRmse']:.2f}\n{100 * best['finalTrainAccuracy']:.1f}% / {100 * best['finalTestAccuracy']:.1f}%"
                        axis.text(x, y, text, ha="center", va="center", fontsize=8)
                artist = axis.imshow(values, cmap="RdYlGn_r", vmin=0, vmax=32, aspect="auto")
                axis.set_xticks(range(len(plan["widths"])), plan["widths"])
                axis.set_yticks(range(len(plan["depths"])), plan["depths"])
                axis.set_xlabel("Width")
                axis.set_ylabel("Hidden depth")
                axis.set_title(f"{operation}/{scale} · {titles[column]}", fontsize=9)
        figure.colorbar(artist, ax=axes, label="Training RMSE in residue units", extend="max", shrink=.7)
        figure.suptitle(label + "Deep-network screens · best final training RMSE per cell\n"
                       "1,000 epochs, seed 11; labels: train RMSE and rounded train / held-out accuracy")
        save(figure, directory, prefix + "-deep-grid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    plot_rotary(directory)
    plot_scalar(directory)
    plot_investigation(directory)


if __name__ == "__main__":
    main()
