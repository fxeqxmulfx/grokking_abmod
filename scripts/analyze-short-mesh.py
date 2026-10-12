#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.10,<4"]
# ///
"""Archive complete fine-resolution mesh trials and compare short grokking plateaus."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grokking-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["svg.hashsalt"] = "short-plateau-mesh"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def longest_plateau(rows, train_threshold, low_threshold, spacing, before):
    start = None
    best = {"epochs": 0, "start": None, "end": None}
    previous = None
    for row in rows:
        epoch = row["epoch"]
        if before is not None and epoch >= before:
            break
        if previous is not None and epoch - previous > spacing + 1e-6:
            start = None
        previous = epoch
        if row["train"] >= train_threshold and row["test"] <= low_threshold:
            start = epoch if start is None else start
            if epoch - start > best["epochs"]:
                best = {"epochs": epoch - start, "start": start, "end": epoch}
        else:
            start = None
    return best


def analyze(directory, minimum, maximum):
    settings = metadata(directory / "run.txt")
    if settings.get("status") != "complete":
        return None
    with (directory / "loss.csv").open(newline="") as source:
        raw = list(csv.DictReader(source))
    rows = [{"epoch": float(row["completed_epochs"]),
             "train": float(row["train_accuracy"]), "test": float(row["validation_accuracy"]),
             "train_loss": float(row["train_cross_entropy"]),
             "test_loss": float(row["validation_cross_entropy"])} for row in raw]
    if not rows or any(not math.isfinite(value) for row in rows for value in row.values()):
        return None
    spacing = int(settings["report_every"]) / int(settings["steps_per_epoch"])
    first = lambda key, threshold: next((row for row in rows if row[key] >= threshold), None)
    high = first("test", .90)
    target = first("test", .99)
    low = next((row for row in reversed(rows) if high and row["epoch"] < high["epoch"] and row["test"] <= .10), None)
    plateau = longest_plateau(rows, .99, .10, spacing, high["epoch"] if high else None)
    perfect = longest_plateau(rows, 1, .10, spacing, high["epoch"] if high else None)
    rise = high["epoch"] - low["epoch"] if high and low else None
    matched = plateau["epochs"] >= minimum and rise is not None and rise <= maximum and target is not None
    stats = {"observations": len(rows), "spacing_epochs": spacing,
             "plateau": plateau, "perfect_train_plateau": perfect,
             "first_train_99": first("train", .99), "last_test_10": low,
             "first_test_90": high, "rise_epochs": rise,
             "first_test_99": target, "final": rows[-1], "matched": matched}
    return {"trial": directory.name, "directory": str(directory.resolve()),
            "settings": settings, "parameters": int(settings["parameters"]), "statistics": stats}, rows


def profile(run):
    s = run["settings"]
    return tuple(s.get(key) for key in ("architecture", "activation", "width", "depth", "mesh_depth",
                                      "connectivity", "learning_rate", "weight_decay", "mesh_learning_rate_multiplier",
                                      "model", "heads", "layers", "attention_mesh_depth", "ffn_mesh_depth",
                                      "embeddings", "mesh_connectivity", "embedding_norm", "initialization_gain",
                                      "embedding_output_gain", "embedding_init_unit", "qk_signs",
                                      "rope", "qk_norm_enabled", "xsa_enabled"))


def title(run):
    s = run["settings"]
    if s.get("model") == "mesh_gpt":
        return f"Mesh Transformer W{s['width']}, L{s['layers']}, H{s['heads']}, A{s['attention_mesh_depth']}, F{s['ffn_mesh_depth']}"
    if s.get("architecture") in ("residual-mesh-mlp", "tied-concat-residual-mesh-mlp", "tied-rotary-residual-mesh-mlp"):
        prefix = {"residual-mesh-mlp": "", "tied-concat-residual-mesh-mlp": "Shared-E concat ", "tied-rotary-residual-mesh-mlp": "Shared-E Rotate90 "}[s["architecture"]]
        return f"{prefix}{s['connectivity'].capitalize()} residual mesh W{s['width']}, B{s['depth']}, M{s['mesh_depth']}, {s['activation']}"
    return f"{s['connectivity'].capitalize()} mesh W{s['width']}, depth {s['depth']}, {s['activation']}"


def check_replay(reference, runs, destination):
    settings = metadata(reference / "run.txt")
    protocol = ("encoding", "operation", "modulus", "train_fraction", "seed", "init_seed", "batch",
                "betas", "epsilon", "warmup_steps", "normalization", "initialization")
    matches = [run for run in runs if profile(run) == profile({"settings": settings})
               and all(run["settings"].get(key) == settings.get(key) for key in protocol)]
    if len(matches) != 1:
        raise ValueError("Replay reference must match exactly one complete trial and its training protocol")
    run = matches[0]

    def observations(path):
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        indexed = {float(row["completed_epochs"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"Duplicate epochs in {path}")
        return indexed

    original = observations(reference / "loss.csv")
    replay = observations(destination / run["history_file"])
    common = sorted(original.keys() & replay.keys())
    if len(common) < 2:
        raise ValueError("Replay reference has fewer than two common observations")
    columns = [key for key in original[common[0]] if key != "elapsed_seconds"]
    differences = {}
    for key in columns:
        pairs = [(float(original[epoch][key]), float(replay[epoch][key])) for epoch in common]
        if any(not math.isfinite(value) for pair in pairs for value in pair):
            raise ValueError(f"Nonfinite replay observation in {key}")
        differences[key] = max(abs(a - b) for a, b in pairs)
    reference_archive = destination / "reference-coarse"
    reference_archive.mkdir(parents=True, exist_ok=True)
    for name in ("loss.csv", "run.txt", "candidate.args", "arguments.json", "outcome.json"):
        source = reference / name
        if source.exists():
            shutil.copyfile(source, reference_archive / name)
    initial_sha = digest(reference / "initial-model.bin")
    result = {"reference_directory": str(reference.resolve()), "replay_trial": run["trial"],
              "reference_history_sha256": digest(reference_archive / "loss.csv"),
              "replay_history_sha256": run["history_sha256"],
              "reference_initial_model_sha256": initial_sha,
              "replay_initial_model_sha256": run["initial_model_sha256"],
              "initial_model_identical": initial_sha == run["initial_model_sha256"],
              "common_observations": len(common), "first_common_epoch": common[0], "last_common_epoch": common[-1],
              "maximum_absolute_difference": differences,
              "scope": "All saved numerical CSV fields at common epochs; elapsed time excluded. CSV precision only."}
    result["passed"] = result["initial_model_identical"] and all(value == 0 for value in differences.values())
    (destination / "replay-check.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise ValueError("Replay differs from its coarse reference; see replay-check.json")
    print(json.dumps({"replay_passed": True, "common_observations": len(common)}))


def save(figure, destination, name):
    for extension in ("png", "svg"):
        path = destination / f"{name}.{extension}"
        figure.savefig(path, dpi=180, metadata={"Date": None} if extension == "svg" else {})
        print(path)
    plt.close(figure)


def plot(groups, destination, detail_seed):
    preferred = [("tied-rotary-residual-mesh-mlp", "square", "128", "neighbor", "rotary-neighbor-short-plateau"),
                 ("tied-rotary-residual-mesh-mlp", "square", "1024", "neighbor", "rotary-neighbor-wide-short-plateau"),
                 ("tied-concat-residual-mesh-mlp", "square", "128", "neighbor", "tied-concat-neighbor-short-plateau"),
                 ("residual-mesh-mlp", "square", "128", "neighbor", "neighbor-mesh-short-plateau"),
                 ("residual-mesh-mlp", "square", "128", "butterfly", "residual-mesh-short-plateau"),
                 ("mesh-mlp", "relu", "384", "butterfly", "relu-mesh-short-plateau"),
                 ("residual-mesh-mlp", "square", "96", "butterfly", None),
                 ("residual-mesh-mlp", "square", "64", "butterfly", None)]
    selected = []
    for model, activation, width, connectivity, filename in preferred:
        values = next((values for key, values in groups.items()
                       if key[:3] == (model, activation, width) and key[5] == connectivity
                       and (not model.startswith("tied-rotary") or key[3:5] == ("2", "3"))), [])
        if values:
            selected.append((values, filename))
    for values in groups.values():
        if not any(values is previous for previous, _ in selected):
            selected.append((values, None))
    if not selected:
        return
    columns = min(2, len(selected))
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(9 if columns == 1 else 12.5, 4.5 * rows),
                               squeeze=False, layout="constrained")
    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)
    for axis, (values, _) in zip(axes.flat, selected):
        colors = plt.cm.tab10.colors
        for index, (run, rows) in enumerate(sorted(values, key=lambda item: int(item[0]["settings"]["seed"]))):
            color = colors[index % len(colors)]
            x = [row["epoch"] for row in rows]
            axis.plot(x, [100 * row["test"] for row in rows], color=color, linewidth=1.6,
                      label=f"seed {run['settings']['seed']}")
            axis.plot(x, [100 * row["train"] for row in rows], color=color, linewidth=.8,
                      linestyle="--", alpha=.65)
        axis.axhline(10, color="gray", linewidth=.6, linestyle=":")
        axis.axhline(90, color="gray", linewidth=.6, linestyle=":")
        axis.set(title=f"{title(values[0][0])}\n{values[0][0]['parameters']:,} parameters",
                 xlabel="Complete training epochs", ylabel="Accuracy (%)", ylim=(0, 101), xlim=(0, None))
        axis.grid(alpha=.2)
        axis.legend(title="Solid: test / dashed: train", fontsize=8, title_fontsize=8, loc="lower right")
    figure.suptitle("Division modulo 97: full-split evaluation every epoch, independent seeds")
    save(figure, destination, "short-mesh-seeds")
    for values, filename in selected:
        if filename:
            run, rows = next((item for item in values if int(item[0]["settings"]["seed"]) == detail_seed), values[0])
            plot_details(run, rows, destination, filename)


def plot_details(run, rows, destination, filename):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    x = [row["epoch"] for row in rows]
    for key, color, label in (("train", "#da7c30", "Train"), ("test", "#3978b5", "Held-out")):
        axes[0].plot(x, [100 * row[key] for row in rows], color=color, label=label)
        axes[1].semilogy(x, [max(1e-10, row[key + "_loss"]) for row in rows], color=color, label=label)
    p = run["statistics"]["plateau"]
    if p["start"] is not None:
        for axis in axes:
            axis.axvspan(p["start"], p["end"], color="gray", alpha=.15,
                         label=f"Train ≥99%, test ≤10%\n{p['epochs']:g} epochs")
    perfect = run["statistics"]["perfect_train_plateau"]
    if perfect["start"] is not None:
        for axis in axes:
            axis.axvspan(perfect["start"], perfect["end"], color="#765b9c", alpha=.18,
                         label=f"Train =100%, test ≤10%\n{perfect['epochs']:g} epochs")
    axes[0].set(ylabel="Accuracy (%)", ylim=(0, 101))
    axes[1].set(ylabel="Cross entropy (log scale)")
    for axis, location in zip(axes, ("lower right", "upper right"), strict=True):
        axis.set(xlabel="Complete training epochs", xlim=(0, rows[-1]["epoch"]))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc=location)
    figure.suptitle(f"{title(run)}, seed {run['settings']['seed']}\nMeasured accuracy, loss, and memorization intervals")
    save(figure, destination, filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plateau-epochs", type=float, default=25)
    parser.add_argument("--jump-epochs", type=float, default=250)
    parser.add_argument("--detail-seed", type=int, default=7)
    parser.add_argument("--replay-reference", type=Path,
                        help="Compare a matching coarse-cadence trial at all shared epochs; fail on any difference")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs, groups = [], {}
    for metadata_path in sorted((args.study / "trials").glob("*/run.txt")):
        result = analyze(metadata_path.parent, args.plateau_epochs, args.jump_epochs)
        if result is None:
            continue
        run, rows = result
        destination = args.output / "raw" / run["trial"]
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("loss.csv", "run.txt", "candidate.args", "arguments.json", "outcome.json"):
            source = metadata_path.parent / name
            if source.exists():
                shutil.copyfile(source, destination / name)
        run["history_file"] = str(Path("raw") / run["trial"] / "loss.csv")
        run["history_sha256"] = digest(destination / "loss.csv")
        run["initial_model_sha256"] = digest(metadata_path.parent / "initial-model.bin")
        run["final_model_sha256"] = digest(metadata_path.parent / "model.bin")
        runs.append(run)
        groups.setdefault(profile(run), []).append((run, rows))
    report = {"protocol": {"minimum_plateau_epochs": args.plateau_epochs,
                           "maximum_rise_epochs": args.jump_epochs, "train_threshold": .99,
                           "low_test_threshold": .10, "high_test_threshold": .90, "final_test_threshold": .99},
              "trainer_sha256": digest(args.study / "bin/mesh-gpt"),
              "analysis_script_sha256": digest(Path(__file__)), "runs": runs}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copyfile(args.study / "policy.txt", args.output / "policy.txt")
    if args.replay_reference:
        check_replay(args.replay_reference, runs, args.output)
    plot(groups, args.output, args.detail_seed)
    print(json.dumps({"completed": len(runs), "matches": sum(r["statistics"]["matched"] for r in runs),
                      "profiles": [{"model": title(values[0][0]), "parameters": values[0][0]["parameters"],
                                    "seeds": [int(r["settings"]["seed"]) for r, _ in values],
                                    "matched_seeds": [int(r["settings"]["seed"]) for r, _ in values if r["statistics"]["matched"]]}
                                   for values in groups.values()]}, indent=2))


if __name__ == "__main__":
    main()
