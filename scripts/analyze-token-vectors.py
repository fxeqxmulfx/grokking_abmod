#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3"]
# ///
"""Measure discrete-log Fourier structure and separation of saved residue vectors."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def primitive_root(prime):
    return next(g for g in range(2, prime) if len({pow(g, k, prime) for k in range(prime - 1)}) == prime - 1)


def concentration(rows, top):
    """Fraction of non-constant Fourier power in the strongest frequencies of an ordered cycle."""
    centered = rows - rows.mean(axis=0, keepdims=True)
    power = (np.abs(np.fft.rfft(centered, axis=0)) ** 2).sum(axis=1)
    # Each frequency below Nyquist also carries its conjugate's equal power.
    power[1:(len(rows) + 1) // 2] *= 2
    power = power[1:]
    order = np.argsort(power)[::-1][:top]
    return float(power[order].sum() / power.sum()), (order + 1).tolist()


def analyze(directory, top, permutations, seed):
    metadata = dict(line.split("=", 1) for line in (directory / "run.txt").read_text().splitlines())
    prime = int(metadata["modulus"])
    with (directory / "embeddings.csv").open(newline="") as source:
        vectors = {row["symbol"]: np.array([float(value) for key, value in row.items() if key.startswith("c")])
                   for row in csv.DictReader(source)}
    residues = np.stack([vectors[str(residue)] for residue in range(prime)])
    # Division acts additively on discrete logarithms, a cycle of length p - 1.
    root = primitive_root(prime)
    cycle = residues[[pow(root, k, prime) for k in range(prime - 1)]]
    observed, frequencies = concentration(cycle, top)
    generator = np.random.default_rng(seed)
    null = np.array([concentration(cycle[generator.permutation(prime - 1)], top)[0]
                     for _ in range(permutations)])
    centered = residues - residues.mean(axis=0, keepdims=True)
    energy = np.linalg.svd(centered, compute_uv=False) ** 2
    energy /= energy.sum()
    unit = residues / np.linalg.norm(residues, axis=1, keepdims=True)
    cosine = unit @ unit.T
    np.fill_diagonal(cosine, -np.inf)
    with (directory / "loss.csv").open(newline="") as source:
        last = list(csv.DictReader(source))[-1]
    return {"run": str(directory.resolve()), "embeddings": metadata["embeddings"],
            "mesh_connectivity": metadata["mesh_connectivity"], "parameters": int(metadata["parameters"]),
            "status": metadata["status"], "stop_reason": metadata.get("stop_reason"),
            "completed_epochs": int(last["completed_epochs"]),
            "train_accuracy": float(last["train_accuracy"]),
            "validation_accuracy": float(last["validation_accuracy"]),
            "embeddings_sha256": hashlib.sha256((directory / "embeddings.csv").read_bytes()).hexdigest(),
            "discrete_log_top_power_fraction": observed, "strongest_frequencies": frequencies,
            "permutation_null_mean": float(null.mean()),
            "permutation_null_p99": float(np.quantile(null, .99)),
            "permutation_p_value": float((1 + (null >= observed).sum()) / (1 + permutations)),
            "effective_rank": float(np.exp(-(energy * np.log(energy, where=energy > 0, out=np.zeros_like(energy))).sum())),
            "directions_for_99_percent_energy": int(np.searchsorted(np.cumsum(energy), .99) + 1),
            "maximum_pair_cosine": float(cosine.max()),
            "mean_nearest_neighbor_cosine": float(cosine.max(axis=1).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = {"protocol": {"order": "discrete_logarithm_of_nonzero_residues", "top_frequencies": arguments.top,
                           "null": "random_cycle_order", "permutations": arguments.permutations,
                           "seed": arguments.seed, "cosine": "between_distinct_residue_vectors"},
              "runs": [analyze(directory, arguments.top, arguments.permutations, arguments.seed)
                       for directory in arguments.directories]}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)
        print(arguments.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
