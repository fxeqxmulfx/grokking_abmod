#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run named Mesh GPT or simple-model candidates with frozen settings and an exact epoch cap."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def mesh_parameters(settings):
    width, layers = settings["width"], settings["layers"]
    attention, ffn = settings["attention-depth"], settings["ffn-depth"]
    # A table ablation stores one vector per residue, operator and equals sign.
    if settings.get("embedding", "mesh") == "table":
        embedding = (settings["modulus"] + 2) * width
    else:
        embedding = 2 * width * attention
    # Count learned weights: a frozen embedding is excluded, and each
    # protected feature fixes two weights in every block's output and FFN mesh.
    learned_embedding = 0 if settings.get("freeze-embedding") else embedding
    protected = 4 * layers * settings.get("preserve-features", 0)
    return learned_embedding + 2 * width * layers * (4 * attention + ffn) - protected


def simple_parameters(settings):
    """Bias-free dense layers of `mesh-gpt grok-simple`, with its defaults."""
    modulus, width = settings["modulus"], settings.get("width", 128)
    features = {"one-hot": modulus, "binary": max(1, (modulus - 1).bit_length()),
                "scalar": 1, "discrete-log": 2}[settings.get("encoding", "one-hot")]
    model, depth = settings.get("model", "mlp"), settings.get("depth", 1)
    if model == "mlp":
        return 2 * features * width + (depth - 1) * width**2 + width * modulus
    if model == "residual-mlp":
        return 2 * features * width + 2 * depth * settings.get("expansion", 4) * width**2 + width * modulus
    if model == "bilinear":
        return 2 * features * width + width * modulus
    raise ValueError(f"unknown simple model: {model}")


PARAMETERS = {"grok": mesh_parameters, "grok-simple": simple_parameters}


def missed(directory, settings):
    """A finished run missed its target if its last complete-split observation is below it."""
    with (directory / "loss.csv").open(newline="") as source:
        last = list(csv.DictReader(source))[-1]
    return float(last["validation_accuracy"]) < settings["stop-validation-accuracy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("labels", nargs="*")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/mesh-gpt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-miss", action="store_true",
                        help="skip the remaining labels once a run misses its validation target")
    arguments = parser.parse_args()
    study = json.loads(arguments.manifest.read_text())
    subcommand = study.get("command", "grok")
    if subcommand not in PARAMETERS:
        parser.error("the manifest command must be grok or grok-simple")
    if type(study["epochs"]) is not int or not 1 <= study["epochs"] <= 10_000:
        parser.error("each candidate must use between 1 and 10,000 full epochs")
    candidates = {candidate["label"]: candidate for candidate in study["candidates"]}
    if len(candidates) != len(study["candidates"]):
        parser.error("candidate labels must be unique")
    if set(arguments.labels) - candidates.keys():
        parser.error("unknown candidate label")
    for label in arguments.labels or candidates:
        if not label or Path(label).name != label or label in (".", ".."):
            parser.error("candidate labels must be plain directory names")
        settings = {**study["common"], **{key: value for key, value in candidates[label].items() if key != "label"}}
        if "steps" in settings or "epochs" in settings or "output" in settings:
            parser.error("budget and output are controlled by the manifest")
        if settings["operation"] != "div" or settings["modulus"] != 97 or settings["train-fraction"] != .5:
            parser.error("this search uses division mod 97 and a fixed half split")
        if arguments.stop_after_miss and "stop-validation-accuracy" not in settings:
            parser.error("--stop-after-miss needs a stop-validation-accuracy setting")
        parameters = PARAMETERS[subcommand](settings)
        directory = ROOT / study["output"] / label
        command = [str(arguments.binary.resolve()), subcommand, "--epochs", str(study["epochs"])]
        for key, value in settings.items():
            # JSON booleans select switches that take no value.
            if value is True:
                command.append(f"--{key}")
            elif value is not False:
                command.extend((f"--{key}", str(value)))
        command.extend(("--output", str(directory)))
        print(json.dumps({"label": label, "parameters": parameters, "epochs": study["epochs"],
                          "command": command}), flush=True)
        if arguments.dry_run:
            continue
        configuration = {"epochs": study["epochs"], "settings": settings}
        if subcommand != "grok":
            configuration["command"] = subcommand
        provenance = directory / "search.json"
        if directory.exists() and any(directory.iterdir()):
            if not provenance.exists() or json.loads(provenance.read_text())["configuration"] != configuration:
                raise FileExistsError(f"Refusing to overwrite another run: {directory}")
            metadata = dict(line.split("=", 1) for line in (directory / "run.txt").read_text().splitlines())
            if metadata.get("status") != "complete":
                raise FileExistsError(f"Incomplete run needs a new attempt label; optimizer state is not saved: {directory}")
            print(f"Already complete: {label}", flush=True)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            provenance.write_text(json.dumps({"configuration": configuration, "command": command,
                                             "binary_sha256": hashlib.sha256(arguments.binary.read_bytes()).hexdigest()},
                                            indent=2) + "\n")
            with (directory / "console.log").open("w") as log:
                with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    if process.wait():
                        raise SystemExit(f"Candidate failed: {label}; artifacts retained in {directory}")
            # Simple models report every weight, so the formula must match exactly.
            if subcommand == "grok-simple":
                metadata = dict(line.split("=", 1) for line in (directory / "run.txt").read_text().splitlines())
                if int(metadata["parameters"]) != parameters:
                    raise SystemExit(f"Parameter count differs from the manifest formula: {label}")
        # A ladder of shrinking candidates ends at its first miss.
        if arguments.stop_after_miss and missed(directory, settings):
            print(f"Missed the validation target; skipping the labels after {label}", flush=True)
            break


if __name__ == "__main__":
    main()
