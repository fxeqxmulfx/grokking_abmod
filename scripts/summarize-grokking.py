#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Compare exhaustive dense and Mesh GPT division runs in complete epochs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mesh_run(directory):
    metadata = dict(line.split("=", 1) for line in (directory / "run.txt").read_text().splitlines())
    count, batch = int(metadata["training_equations"]), int(metadata["batch"])
    updates = math.ceil(count / batch)
    with (directory / "loss.csv").open(newline="") as source:
        raw = list(csv.DictReader(source))
    history = []
    for row in raw:
        step = int(row["step"])
        complete, partial = divmod(step, updates)
        seen = int(row.get("training_examples_seen", complete * count + min(partial * batch, count)))
        history.append({"step": step, "epoch": seen / count,
                        "train": float(row["train_accuracy"]), "heldout": float(row["validation_accuracy"])})
    return {"family": "mesh", "parameters": int(metadata["parameters"]),
            "status": metadata["status"], "settings": metadata, "history": history,
            "history_sha256": digest(directory / "loss.csv")}


def dense_run(directory):
    manifest = json.loads((directory / "experiment.json").read_text())
    settings = manifest["description"]
    model, benchmark = settings["model"], settings["benchmark"]
    block, attention = model["block"], model["block"]["attention"]
    # Exact count for this study's bias-free, parameter-free RMSNorm GPTMini.
    if not (benchmark["type"] == "ModularDivision"
            and attention["projections"] == {"type": "FusedQKV", "bias": False}
            and attention["scores"]["type"] == "QKNorm"
            and not attention["output_bias"] and not block["ffn"]["bias"]
            and block["norm"]["type"] == "RMSNorm" and not block["norm"]["scale"]
            and model["final_norm"]["type"] == "RMSNorm" and not model["final_norm"]["scale"]
            and model["readout"]["type"] == "Tied" and model["positions"]["type"] == "RoPE"):
        raise ValueError(f"Unsupported dense parameter-count protocol: {directory}")
    width, layers = model["width"], model["depth"]
    vocab = benchmark["prime"] + 142  # Author vocabulary: EOS, equals, 20 operators, 120 permutations.
    parameters = width * vocab + layers * ((4 + 2 * block["ffn"]["multiplier"]) * width**2 + attention["heads"])
    raw = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    history = [{"step": row["step"], "epoch": row["epochs_seen"],
                "train": row["train"]["accuracy"], "heldout": row["heldout"]["accuracy"]} for row in raw]
    return {"family": "dense", "parameters": parameters,
            "status": "complete" if (directory / "result.json").exists() else "running",
            "settings": settings, "history": history, "history_sha256": digest(directory / "history.jsonl")}


def streaks(history, predicate, length):
    """Every first confirmation in a consecutive qualifying stretch."""
    start, count = None, 0
    for row in history:
        if predicate(row):
            if not count:
                start = row
            count += 1
            if count == length:
                yield {"onset_step": start["step"], "onset_epoch": start["epoch"],
                       "confirmed_step": row["step"], "confirmed_epoch": row["epoch"]}
        else:
            count = 0


def summarize(directory, archive=None):
    run = mesh_run(directory) if (directory / "run.txt").exists() else dense_run(directory)
    history = run.pop("history")
    if not history or any(not 0 <= row[key] <= 1 for row in history for key in ("train", "heldout")):
        raise ValueError(f"Missing history or invalid accuracy: {directory}")
    if any(a["step"] >= b["step"] or a["epoch"] >= b["epoch"] for a, b in zip(history, history[1:])):
        raise ValueError(f"Non-increasing history: {directory}")
    if history[-1]["epoch"] > 10_000:
        raise ValueError(f"Run exceeded the 10,000-epoch cap: {directory}")
    if archive:
        relative = Path("histories") / f"{run['family']}-{directory.name}.csv"
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=("step", "epoch", "train", "heldout"), lineterminator="\n")
            writer.writeheader()
            writer.writerows(history)
        run["history_file"] = str(relative)
        run["plot_history_sha256"] = digest(destination)
    plateau = list(streaks(history, lambda row: row["train"] >= .99 and row["heldout"] <= .1, 5))
    # Generalization is the first observation at the held-out target; runs may stop there.
    event = next(({"step": row["step"], "epoch": row["epoch"]}
                  for row in history if row["heldout"] >= .99), None)
    grok = event if event and any(p["confirmed_step"] < event["step"] for p in plateau) else None
    last = history[-1]
    first_train = next((row["epoch"] for row in history if row["train"] >= .99), None)
    return {"run": str(directory.resolve()), **run, "observations": len(history),
            "epochs": last["epoch"], "first_train_99_epoch": first_train,
            "generalization": event,
            "generalization_lag_epochs": event["epoch"] - first_train if event and first_train is not None else None,
            "delayed_grokking": grok,
            "memorization_plateau": next(iter(plateau), None),
            "final_train_accuracy": last["train"], "final_heldout_accuracy": last["heldout"],
            "peak_heldout_accuracy": max(row["heldout"] for row in history)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    archive = arguments.output.parent if arguments.output else None
    runs = [summarize(directory, archive) for directory in arguments.directories]
    report = {"protocol": {"target": .99, "generalization": "first_heldout_observation_at_target",
                           "plateau_observations": 5, "plateau_heldout_ceiling": .1,
                           "epoch_cap": 10_000}, "runs": runs}
    report["smallest_tested_generalizing"] = {
        family: min((run["parameters"] for run in runs
                     if run["family"] == family and run["generalization"]), default=None)
        for family in sorted({run["family"] for run in runs})}
    report["smallest_tested_delayed_grokking"] = {
        family: min((run["parameters"] for run in runs
                     if run["family"] == family and run["delayed_grokking"]), default=None)
        for family in sorted({run["family"] for run in runs})}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)
        print(arguments.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
