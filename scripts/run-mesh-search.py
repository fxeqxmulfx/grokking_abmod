#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run named Mesh GPT candidates with frozen settings and an exact epoch cap."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("labels", nargs="*")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/mesh-gpt")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    study = json.loads(arguments.manifest.read_text())
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
        width, layers = settings["width"], settings["layers"]
        attention, ffn = settings["attention-depth"], settings["ffn-depth"]
        # A table ablation stores one vector per residue, operator and equals sign.
        if settings.get("embedding", "mesh") == "table":
            embedding = (settings["modulus"] + 2) * width
        else:
            embedding = 2 * width * attention
        parameters = embedding + 2 * width * layers * (4 * attention + ffn)
        directory = ROOT / study["output"] / label
        command = [str(arguments.binary.resolve()), "grok", "--epochs", str(study["epochs"])]
        for key, value in settings.items():
            command.extend((f"--{key}", str(value)))
        command.extend(("--output", str(directory)))
        print(json.dumps({"label": label, "parameters": parameters, "epochs": study["epochs"],
                          "command": command}), flush=True)
        if arguments.dry_run:
            continue
        configuration = {"epochs": study["epochs"], "settings": settings}
        provenance = directory / "search.json"
        if directory.exists() and any(directory.iterdir()):
            if not provenance.exists() or json.loads(provenance.read_text())["configuration"] != configuration:
                raise FileExistsError(f"Refusing to overwrite another run: {directory}")
            metadata = dict(line.split("=", 1) for line in (directory / "run.txt").read_text().splitlines())
            if metadata.get("status") != "complete":
                raise FileExistsError(f"Incomplete run needs a new attempt label; optimizer state is not saved: {directory}")
            print(f"Already complete: {label}", flush=True)
            continue
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


if __name__ == "__main__":
    main()
