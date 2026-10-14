# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.10.0", "numpy>=2.2,<3"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Rotate90 Dense + ReLU. Run: uv run rotate90.py --epochs 10000.

E is shared by both operands and the readout. Only E and A are learned.
This file recreates the selected experiment from its seed. Initialization,
split and minibatch order match the browser; floating-point training differs
between PyTorch, WASM and Rust/CUDA. Metrics are written as CSV to stdout.
Original grokking paper: https://arxiv.org/abs/2201.02177
"""

import argparse
import csv
import math
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_WIDTH = 128
DEFAULT_SEED = 11
DEFAULT_EPOCHS = 10000
P = 97
TRAIN = P * (P - 1) // 2
BATCH = 512
EPSILON = 1e-5
MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return value ^ (value >> 31)

    def unit(self) -> float:
        return (self.next() >> 40) / 16777216

    def index(self, bound: int) -> int:
        threshold = (1 << 64) % bound
        while True:
            value = self.next()
            if value >= threshold:
                return value % bound

    def shuffle(self, values: list) -> None:
        for end in range(len(values) - 1, 0, -1):
            index = self.index(end + 1)
            values[end], values[index] = values[index], values[end]


def initial_weights(width: int, seed: int) -> torch.Tensor:
    random = SplitMix64(seed ^ 0x73696D706C650004)
    values = []
    for count, variance in [(P * width, 1 / width), (width * width, 2 / width)]:
        deviation = math.sqrt(variance)
        for _ in range(0, count, 2):
            radius = math.sqrt(-2 * math.log(1 - random.unit()))
            angle = 2 * math.pi * random.unit()
            values.extend((deviation * radius * math.cos(angle),
                           deviation * radius * math.sin(angle)))
    return torch.tensor(values, dtype=torch.float32)


def dataset(seed: int) -> torch.Tensor:
    rows = [(a, b, a * pow(b, -1, P) % P)
            for a in range(P) for b in range(1, P)]
    SplitMix64(seed ^ 0x73706C6974730002).shuffle(rows)
    return torch.tensor(rows, dtype=torch.int64)


class Sampler:
    def __init__(self, seed: int):
        self.random = SplitMix64(seed ^ 0x6261746368657303)
        self.order = list(range(TRAIN))
        self.random.shuffle(self.order)
        self.position = 0

    def sample(self) -> list[int]:
        if self.position == TRAIN:
            self.random.shuffle(self.order)
            self.position = 0
        start = self.position
        self.position = min(TRAIN, start + BATCH)
        return self.order[start:self.position]


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPSILON)


class Rotate90(nn.Module):
    def __init__(self, width: int = DEFAULT_WIDTH, seed: int = DEFAULT_SEED):
        super().__init__()
        if width < 2 or width % 2:
            raise ValueError("Width must be a positive even integer of at least 2.")
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("Seed must be between 0 and 4294967295.")
        weights = initial_weights(width, seed)
        self.E = nn.Parameter(weights[:P * width].reshape(P, width))
        self.A = nn.Parameter(weights[P * width:].reshape(width, width))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pairs = self.E[b].reshape(*b.shape, -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        x = self.E[a] + rotated
        x = x + F.relu(rms_norm(x) @ self.A)
        return rms_norm(x) @ self.E.T


@torch.no_grad()
def evaluate(model: Rotate90, rows: torch.Tensor) -> tuple[float, float]:
    loss, correct = 0.0, 0
    for batch in rows.split(BATCH):
        logits = model(batch[:, 0], batch[:, 1])
        loss += F.cross_entropy(logits, batch[:, 2], reduction="sum").item()
        correct += (logits.argmax(-1) == batch[:, 2]).sum().item()
    return loss / len(rows), correct / len(rows)


def train(width: int, seed: int, epochs: int, continue_after_target: bool = False) -> Rotate90:
    model = Rotate90(width, seed)
    rows = dataset(seed)
    sampler = Sampler(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                 betas=(0.9, 0.98), eps=1e-8,
                                 weight_decay=0.1, foreach=False)
    writer = csv.writer(sys.stdout)
    writer.writerow(("epoch", "step", "train_loss", "train_accuracy", "test_loss", "test_accuracy"))
    step, target_reached = 0, False
    for epoch in range(epochs + 1):
        if epoch:
            for _ in range(math.ceil(TRAIN / BATCH)):
                batch = rows[sampler.sample()]
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(batch[:, 0], batch[:, 1]), batch[:, 2]).backward()
                step += 1
                optimizer.param_groups[0]["lr"] = 0.001 * min(step / 10, 1)
                optimizer.step()
        train_loss, train_accuracy = evaluate(model, rows[:TRAIN])
        test_loss, test_accuracy = evaluate(model, rows[TRAIN:])
        writer.writerow((epoch, step, train_loss, train_accuracy, test_loss, test_accuracy))
        sys.stdout.flush()
        if not target_reached and test_accuracy >= 0.99:
            target_reached = True
            print("Paused at 99% test accuracy.", file=sys.stderr)
            if not continue_after_target:
                if not sys.stdin.isatty():
                    break
                print("Continue training with the same optimizer? [y/N] ", end="", file=sys.stderr, flush=True)
                if input().strip().lower() not in {"y", "yes"}:
                    break
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--continue-after-target", action="store_true",
                        help="Keep training after the first 99%% test accuracy crossing.")
    args = parser.parse_args()
    if args.width < 2 or args.width % 2 or not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("Width must be even and at least 2; seed must be an unsigned 32-bit integer.")
    if not 1 <= args.epochs <= 10000 or args.threads < 1:
        parser.error("Epochs must be between 1 and 10000; threads must be at least 1.")
    torch.set_num_threads(args.threads)
    train(args.width, args.seed, args.epochs, args.continue_after_target)


if __name__ == "__main__":
    main()
