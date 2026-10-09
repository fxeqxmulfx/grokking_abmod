#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.10,<4", "numpy>=2,<3"]
# ///
"""Plot training artifacts from the Mesh Neural Net example (NumPy + Matplotlib)."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("target/mesh-net"))
    directory = parser.parse_args().directory
    loss = np.atleast_1d(np.genfromtxt(directory / "loss.csv", delimiter=",", names=True))
    curve = np.genfromtxt(directory / "predictions.csv", delimiter=",", names=True)
    with (directory / "nodes.csv").open() as source:
        nodes = list(csv.DictReader(source))

    figure, (fit, progress, mesh) = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    fit.plot(curve["x"], curve["target"], color="black", linestyle="--", label="Target")
    fit.plot(curve["x"], curve["initial"], alpha=0.55, label="Initial")
    fit.plot(curve["x"], curve["prediction"], color="tab:orange", label="Trained")
    fit.set(title="Piecewise-function regression", xlabel="Input x", ylabel="Output", ylim=(-0.3, 1.3))
    fit.legend()
    progress.semilogy(loss["epoch"], np.maximum(loss["mse"], 1e-15), color="tab:orange")
    progress.set(title="Full-batch Adam on GPU", xlabel="Epoch", ylabel="Training MSE")
    progress.grid(alpha=0.3)

    positions = {0: (0.0, 0.0)}
    for node in nodes:
        positions[int(node["node"])] = (
            int(node["position"]) - (int(node["width"]) - 1) / 2,
            -int(node["layer"]),
        )
    largest = max(abs(float(node[f"{side}_weight"])) for node in nodes for side in ("left", "right"))
    scale = Normalize(-max(largest, 1e-8), max(largest, 1e-8))
    colors = plt.get_cmap("RdBu_r")
    for node in nodes:
        destination = positions[int(node["node"])]
        for side in ("left", "right"):
            parent = int(node[side])
            if parent < 0:
                continue
            source = positions[parent]
            mesh.plot((source[0], destination[0]), (source[1], destination[1]),
                      color=colors(scale(float(node[f"{side}_weight"]))), linewidth=1.5, zorder=1)
    x, y = zip(*positions.values())
    mesh.scatter(x, y, s=30, facecolor="white", edgecolor="0.3", zorder=2)
    depth = max(int(node["layer"]) for node in nodes)
    mesh.set(title=f"ELU diamond: {depth} layers, {len(nodes)} neurons", aspect="equal")
    mesh.axis("off")
    figure.colorbar(plt.cm.ScalarMappable(norm=scale, cmap=colors), ax=mesh, label="Edge weight", shrink=0.7)
    for extension in ("svg", "png"):
        output = directory / f"mesh-training.{extension}"
        figure.savefig(output, dpi=180)
        print(output)
    plt.close(figure)


if __name__ == "__main__":
    main()
