#!/usr/bin/env python3
"""Plot Mesh GPT next-token cross-entropy and held-out character perplexity."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("target/mesh-gpt"))
    directory = parser.parse_args().directory
    history = np.atleast_1d(np.genfromtxt(directory / "loss.csv", delimiter=",", names=True))
    figure, (loss, perplexity) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    loss.plot(history["step"], history["train_cross_entropy"], label="Training interval mean")
    loss.plot(history["step"], history["validation_cross_entropy"], label="Held-out validation")
    loss.set(title="Mesh GPT next-token prediction", xlabel="AdamW step", ylabel="Cross-entropy (nats / character)")
    loss.legend()
    perplexity.plot(history["step"], history["validation_perplexity"], color="tab:orange")
    perplexity.set(title="Held-out character perplexity", xlabel="AdamW step", ylabel="Perplexity")
    for axis in (loss, perplexity):
        axis.grid(alpha=0.25)
    for extension in ("svg", "png"):
        path = directory / f"gpt-training.{extension}"
        figure.savefig(path, dpi=180)
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
