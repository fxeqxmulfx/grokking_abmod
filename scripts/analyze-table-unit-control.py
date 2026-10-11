#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.10,<4"]
# ///
"""Audit paired table-normalization trials and plot their archived histories."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"default": "Ordinary table", "init-unit": "Unit at initialization", "unit": "Unit every forward"}
COLORS = {"default": "#2563eb", "init-unit": "#d97706", "unit": "#16a34a"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def vectors(path):
    return [[float(value) for key, value in row.items() if key.startswith("c")]
            for row in read_csv(path)]


def norm_stats(rows):
    norms = [math.sqrt(sum(value ** 2 for value in row)) for row in rows]
    return {"mean": statistics.mean(norms), "min": min(norms), "max": max(norms),
            "std": statistics.pstdev(norms)}


def initial_parameters(path):
    data = path.read_bytes()
    if data[:8] not in (b"MSHGPT08", b"MSHGPT09"):
        raise ValueError(f"Expected a table checkpoint: {path}")
    vocab, _, width, heads, layers, attention, ffn, embedding, connectivity = struct.unpack_from("<9I", data, 8)
    if embedding != 1 or connectivity != 0:
        raise ValueError(f"Expected table embeddings and neighbor meshes: {path}")
    offset = 8 + 9 * 4 + (14 * 4 if data[:8] == b"MSHGPT09" else 0) + vocab * 4
    count = vocab * width + 2 * width * layers * (4 * attention + ffn)
    if len(data) != offset + count * 4:
        raise ValueError(f"Unexpected checkpoint length: {path}")
    parameters = struct.unpack_from(f"<{count}f", data, offset)
    rows = [parameters[token * width:(token + 1) * width] for token in range(vocab)]
    return {"shape": [vocab, width, heads, layers, attention, ffn], "rows": rows,
            "table_sha256": hashlib.sha256(data[offset:offset + vocab * width * 4]).hexdigest(),
            "blocks_sha256": hashlib.sha256(data[offset + vocab * width * 4:]).hexdigest()}


def audit(summary, summary_path, manifest, execution=None):
    candidates = {candidate["label"]: candidate for candidate in manifest["candidates"]}
    expected_status = {label: "complete" for label in candidates}
    if execution is not None:
        partitions = [execution[key] for key in ("completed_labels", "interrupted_labels", "not_started_labels")]
        labels = [label for partition in partitions for label in partition]
        if len(labels) != len(set(labels)) or set(labels) != set(candidates):
            raise ValueError("Execution record must partition every predefined candidate exactly once")
        if execution["reason"] != "user_requested_stop":
            raise ValueError("Unsupported interruption reason")
        expected_status = {**{label: "complete" for label in partitions[0]},
                           **{label: "interrupted" for label in partitions[1]}}
    if (len(summary["runs"]) != len(expected_status)
            or {Path(run["run"]).name for run in summary["runs"]} != set(expected_status)):
        raise ValueError("Summary does not contain exactly the executed candidates")
    binary_hashes, trials, grouped = set(), [], {}
    for run in summary["runs"]:
        directory = Path(run["run"])
        settings = run["settings"]
        provenance = json.loads((directory / "search.json").read_text())
        candidate = candidates[directory.name]
        expected = {"epochs": manifest["epochs"], "settings": {
            **manifest["common"], **{key: value for key, value in candidate.items() if key != "label"}}}
        if provenance["configuration"] != expected:
            raise ValueError(f"Trial configuration differs from the predefined manifest: {directory}")
        binary_hashes.add(provenance["binary_sha256"])
        variant = ("init-unit" if settings["embedding_init_unit"] == "true" else
                   "unit" if settings["embedding_norm"] == "unit" else "default")
        seed = int(settings["seed"])
        if (seed != candidate["seed"] or settings["embedding_norm"] != candidate["embedding-norm"]
                or (settings["embedding_init_unit"] == "true") != candidate["embedding-init-unit"]):
            raise ValueError(f"Recorded normalization policy differs from the manifest: {directory}")
        if run["status"] != expected_status[directory.name] or run["parameters"] != 37248:
            raise ValueError(f"Unexpected trial status or parameter count: {directory}")
        complete = run["status"] == "complete"
        archived = summary_path.parent / run["history_file"]
        if digest(archived) != run["plot_history_sha256"] or digest(directory / "loss.csv") != run["history_sha256"]:
            raise ValueError(f"History hash mismatch: {directory}")
        history = read_csv(archived)
        event = next((float(row["epoch"]) for row in history if float(row["heldout"]) >= .99), None)
        if event is not None and event != run["epochs"]:
            raise ValueError(f"Trial continued past its stopping observation: {directory}")
        expected_stop = ("validation_accuracy" if event is not None else "update_budget") if complete else "user_requested"
        if (settings["stop_reason"] != expected_stop
                or (complete and event is None and run["epochs"] != manifest["epochs"])
                or (not complete and (event is not None or run["epochs"] >= manifest["epochs"]))):
            raise ValueError(f"Trial did not reach its target or exhaust the predefined budget: {directory}")
        if not complete:
            observation = execution["interrupted_last_observation"]
            if (observation["label"] != directory.name or observation["epoch"] != run["epochs"]
                    or observation["step"] != int(history[-1]["step"])
                    or observation["train_accuracy"] != run["final_train_accuracy"]
                    or observation["validation_accuracy"] != run["final_heldout_accuracy"]
                    or observation["peak_validation_accuracy"] != run["peak_heldout_accuracy"]
                    or observation["model_checkpoint_sha256"] != digest(directory / "model.bin")):
                raise ValueError(f"Interrupted trial no longer matches its execution record: {directory}")
        if complete:
            predictions = read_csv(directory / "predictions.csv")
            for split, key in [("train", "final_train_accuracy"), ("validation", "final_heldout_accuracy")]:
                rows = [row for row in predictions if row["split"] == split]
                if len(rows) != 4656 or abs(sum(int(row["correct"]) for row in rows) / len(rows) - run[key]) > 1e-8:
                    raise ValueError(f"Prediction/accuracy mismatch: {directory}, {split}")
        initial = initial_parameters(directory / "initial-model.bin")
        effective = vectors(directory / "initial-embeddings.csv")
        trial = {"label": directory.name, "variant": variant, "seed": seed, "status": run["status"],
                 "right_censored": not complete,
                 "epochs_to_99": event, "epochs_run": run["epochs"],
                 "first_train_99_epoch": run["first_train_99_epoch"],
                 "last_observed_train_accuracy": run["final_train_accuracy"],
                 "last_observed_validation_accuracy": run["final_heldout_accuracy"],
                 "peak_observed_validation_accuracy": run["peak_heldout_accuracy"],
                 "final_train_accuracy": run["final_train_accuracy"] if complete else None,
                 "final_validation_accuracy": run["final_heldout_accuracy"] if complete else None,
                 "final_predictions_checked": complete,
                 "stop_reason": settings["stop_reason"],
                 "initial_raw_norms_all_tokens": norm_stats(initial["rows"]),
                 "initial_effective_norms_all_tokens": norm_stats(effective),
                 "final_effective_norms_all_tokens": norm_stats(vectors(directory / "embeddings.csv")) if complete else None,
                 "initial_table_sha256": initial["table_sha256"],
                 "initial_blocks_sha256": initial["blocks_sha256"],
                 "equations_sha256": digest(directory / "equations.csv"),
                 "symbols_sha256": digest(directory / "symbols.csv")}
        trials.append(trial)
        grouped.setdefault(seed, {})[variant] = (trial, initial, effective, read_csv(directory / "loss.csv")[0])
    expected_seeds = {candidates[label]["seed"] for label in expected_status}
    if len(binary_hashes) != 1 or set(grouped) != expected_seeds:
        raise ValueError("All trials must use one binary and every executed seed")
    paired_complete_seeds = [seed for seed, group in sorted(grouped.items())
                             if set(group) == set(manifest["variants"])
                             and all(entry[0]["status"] == "complete" for entry in group.values())]
    if not paired_complete_seeds:
        raise ValueError("No complete seed pairs to aggregate")
    checks = []
    for seed, group in sorted(grouped.items()):
        if set(group) != set(manifest["variants"]):
            continue
        default, once, continuous = (group[name] for name in ["default", "init-unit", "unit"])
        for key in ["equations_sha256", "symbols_sha256", "initial_blocks_sha256"]:
            if len({entry[0][key] for entry in group.values()}) != 1:
                raise ValueError(f"Paired trial mismatch: seed {seed}, {key}")
        if default[1]["shape"] != once[1]["shape"] or default[1]["shape"] != continuous[1]["shape"]:
            raise ValueError(f"Paired model shapes differ: seed {seed}")
        if default[0]["initial_table_sha256"] != continuous[0]["initial_table_sha256"]:
            raise ValueError(f"Ordinary/unit initial raw tables differ: seed {seed}")
        normalized = []
        for row in default[1]["rows"]:
            norm = math.sqrt(sum(value * value for value in row))
            normalized.append([value / norm for value in row])
        direction_error = max(abs(x - y) for a, b in zip(normalized, once[1]["rows"], strict=True)
                              for x, y in zip(a, b, strict=True))
        effective_error = max(abs(x - y) for a, b in zip(once[2], continuous[2], strict=True)
                              for x, y in zip(a, b, strict=True))
        loss_error = max(abs(float(once[3][key]) - float(continuous[3][key]))
                         for key in ["train_cross_entropy", "validation_cross_entropy"])
        if direction_error > 1e-6 or effective_error > 1e-6 or loss_error > 1e-4:
            raise ValueError(f"Initial unit variants do not start from matching functions: seed {seed}")
        checks.append({"seed": seed, "identical_equations_symbols_and_initial_blocks": True,
                       "identical_default_and_unit_raw_table": True,
                       "init_unit_direction_max_error": direction_error,
                       "unit_variants_initial_effective_max_error": effective_error,
                       "unit_variants_initial_loss_max_error": loss_error})
    groups = {}
    for variant in manifest["variants"]:
        subset = [trial for trial in trials if trial["variant"] == variant]
        paired = [trial for trial in subset if trial["seed"] in paired_complete_seeds]
        successful = [trial["epochs_to_99"] for trial in paired if trial["epochs_to_99"] is not None]
        capped = [trial["epochs_to_99"] if trial["epochs_to_99"] is not None else manifest["epochs"]
                  for trial in paired]
        groups[variant] = {"observed_runs": len(subset),
                           "completed_runs": sum(trial["status"] == "complete" for trial in subset),
                           "right_censored_runs": sum(trial["right_censored"] for trial in subset),
                           "reached_99": sum(trial["epochs_to_99"] is not None for trial in subset),
                           "epochs_to_99_by_seed": {str(trial["seed"]): trial["epochs_to_99"] for trial in subset},
                           "right_censored_epoch_by_seed": {str(trial["seed"]): trial["epochs_run"]
                                                            for trial in subset if trial["right_censored"]},
                           "complete_pair_runs": len(paired), "complete_pair_reached_99": len(successful),
                           "median_epochs_to_target_or_cap_on_complete_pairs": statistics.median(capped),
                           "successful_epochs_min_on_complete_pairs": min(successful) if successful else None,
                           "successful_epochs_max_on_complete_pairs": max(successful) if successful else None}
    return {"protocol": {"planned_seeds": manifest["seeds"], "observed_seeds": sorted(grouped),
                         "aggregate_seeds": paired_complete_seeds, "variants": manifest["variants"],
                         "target": .99, "epoch_cap": manifest["epochs"], "observation_interval_epochs": 25,
                         "failed_runs_use_cap_in_group_median": True,
                         "interrupted_runs_excluded_from_group_median": True,
                         "unstarted_labels": execution["not_started_labels"] if execution else [],
                         "stopping_rule": "first_complete_validation_observation_at_target"},
            "binary_sha256": next(iter(binary_hashes)), "source_summary_sha256": digest(summary_path),
            "paired_initialization_checks": checks, "groups": groups,
            "trials": sorted(trials, key=lambda trial: (trial["seed"], manifest["variants"].index(trial["variant"])))}


def plot(summary, report, archive, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams["svg.hashsalt"] = "table-unit-control"
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    trial_by_label = {trial["label"]: trial for trial in report["trials"]}
    for run in summary["runs"]:
        trial = trial_by_label[Path(run["run"]).name]
        history = read_csv(archive / run["history_file"])
        axes[0].plot([float(row["epoch"]) for row in history],
                     [100 * float(row["heldout"]) for row in history],
                     color=COLORS[trial["variant"]], alpha=.55, linewidth=1.2)
    axes[0].axhline(99, color="gray", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="Complete training epochs", ylabel="Validation accuracy (%)", ylim=(0, 101),
                title=f"Complete validation split; {len(report['protocol']['observed_seeds'])} started seeds")
    if max(run["epochs"] for run in summary["runs"]) > 1000:
        axes[0].set_xscale("symlog", linthresh=100)
    axes[0].set_xlim(left=0)
    axes[0].legend(handles=[Line2D([0], [0], color=COLORS[key], label=NAMES[key])
                            for key in report["protocol"]["variants"]], fontsize=8, loc="lower right")
    seeds = report["protocol"]["observed_seeds"]
    for index, seed in enumerate(seeds):
        trials = [trial for trial in report["trials"] if trial["seed"] == seed]
        x = [report["protocol"]["variants"].index(trial["variant"]) + (index - (len(seeds) - 1) / 2) * .025
             for trial in trials]
        y = [trial["epochs_to_99"] if trial["epochs_to_99"] is not None else
             trial["epochs_run"] if trial["right_censored"] else report["protocol"]["epoch_cap"]
             for trial in trials]
        axes[1].plot(x, y, color="gray", alpha=.4, linewidth=.8)
        for position, epochs, trial in zip(x, y, trials, strict=True):
            axes[1].scatter(position, epochs, color=COLORS[trial["variant"]], s=35,
                            marker="o" if trial["epochs_to_99"] is not None else "^" if trial["right_censored"] else "x")
            if trial["right_censored"]:
                axes[1].annotate(f"> {epochs:g} (seed {seed})", (position, epochs),
                                 xytext=(-5, -15), textcoords="offset points", ha="right", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set(xticks=range(3), xticklabels=["Ordinary\ntable", "Unit at\ninitialization", "Unit every\nforward"],
                ylabel="Epochs to first 99% validation",
                title="Lines connect trials with the same seed")
    if any(trial["right_censored"] for trial in report["trials"]):
        axes[1].legend(handles=[Line2D([0], [0], color="gray", marker="^", linestyle="",
                                     label="Interrupted: observed lower bound")], fontsize=8, loc="upper left")
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle("Division modulo 97: initial scale vs. continuous unit normalization")
    for extension in ("svg", "png"):
        path = output.parent / f"table-unit-control.{extension}"
        figure.savefig(path, dpi=180, metadata={"Date": None} if extension == "svg" else {})
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/div97-table-unit-control.json")
    parser.add_argument("--execution", type=Path, help="Record of a user-requested early stop")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = json.loads(arguments.summary.read_text())
    manifest = json.loads(arguments.manifest.read_text())
    execution = json.loads(arguments.execution.read_text()) if arguments.execution else None
    if execution and execution["manifest_sha256"] != digest(arguments.manifest):
        raise ValueError("Execution record refers to a different predefined manifest")
    report = audit(summary, arguments.summary, manifest, execution)
    if execution:
        report["execution_sha256"] = digest(arguments.execution)
    report["manifest_sha256"] = digest(arguments.manifest)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(arguments.output)
    plot(summary, report, arguments.summary.parent, arguments.output)


if __name__ == "__main__":
    main()
