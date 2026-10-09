#!/usr/bin/env python3
"""Plot complete-split Mesh GPT grokking curves and generated token vectors."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(figure, directory, name):
    for extension in ("svg", "png"):
        path = directory / f"{name}.{extension}"
        figure.savefig(path, dpi=180)
        print(path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("target/mesh-gpt-grokking"))
    directory = parser.parse_args().directory
    history = np.atleast_1d(np.genfromtxt(directory / "loss.csv", delimiter=",", names=True))
    figure, (accuracy, loss) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for split, label in (("train", "Training"), ("validation", "Held-out validation")):
        accuracy.plot(history["step"], 100 * history[f"{split}_accuracy"], label=label)
        loss.plot(history["step"], history[f"{split}_cross_entropy"], label=label)
    accuracy.set(title="Answer accuracy on complete splits", ylabel="Correct answers (%)", ylim=(0, 101))
    loss.set(title="Answer-only cross-entropy", ylabel="Cross-entropy (nats / equation)")
    for axis in (accuracy, loss):
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlabel("AdamW update (linear through 10, then log)")
        axis.grid(alpha=0.25)
        axis.legend()
    save(figure, directory, "grokking-training")

    with (directory / "embeddings.csv").open(newline="") as file:
        records = list(csv.DictReader(file))
    residues = sorted((row for row in records if row["symbol"].isdigit()), key=lambda row: int(row["symbol"]))
    columns = [name for name in records[0] if name.startswith("c")]
    vectors = np.array([[float(row[name]) for name in columns] for row in residues])
    centered = vectors - vectors.mean(axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    points = centered @ basis[:2].T
    modulus = len(residues)
    figure, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    for a in range(modulus):
        pair = points[[a, (a + 8) % modulus]]
        axis.plot(pair[:, 0], pair[:, 1], color="gray", alpha=0.25, linewidth=0.7, zorder=0)
    scatter = axis.scatter(points[:, 0], points[:, 1], c=np.arange(modulus) % 8, cmap="tab10", vmin=0, vmax=9, s=24)
    figure.colorbar(scatter, ax=axis, ticks=range(8), label="Residue modulo 8")
    axis.set(title=f"Shared mesh token vectors: PCA, edges a → a+8 mod {modulus}", xlabel="Principal component 1", ylabel="Principal component 2")
    save(figure, directory, "grokking-embeddings")


if __name__ == "__main__":
    main()
