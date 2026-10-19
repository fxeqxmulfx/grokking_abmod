#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = ["torch==2.14.1", "numpy==2.5.4"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# ///
"""Compare the browser Rotate90 model with the transformer's actual SwiGLU.

Run from the repository root using uv run --locked scripts/compare-rotate90.py.
Set ROTATE90_TRANSFORMER_ROOT to the local transformer checkout if needed.
Training writes full histories and checkpoints under target/rotate90-swiglu.
Use --verify to audit an existing output directory without retraining.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("ROTATE90_TRANSFORMER_ROOT", str(Path.home() / "lean_projects/transformer"))).resolve()
for source_root in (LIBRARY / "python/src", LIBRARY / "optimizers/src"):
    sys.path.insert(0, str(source_root))

import numpy as np
import torch
from torch.nn import functional as F
import lean_optim
from lean_optim import ScheduleFreeAdamW
from lab.domain.model import ReLU
from lab.domain.rotary import DenseBranch, Rotate90 as ModelSpec, SwiGLU
from lab.infrastructure.nn.rotary import Rotate90 as LabRotate90

OUTPUT = ROOT / "target/rotate90-swiglu"
REFERENCE = ROOT / "web/own-channel/python/rotate90.py"
spec = importlib.util.spec_from_file_location("rotate90_reference", REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

def sha(data):
    """Hash bytes using the artifact convention of the browser studies."""
    return hashlib.sha256(data).hexdigest()


def tensor_bytes(value):
    """Return a contiguous CPU tensor's exact bytes."""
    return value.detach().cpu().contiguous().numpy().tobytes()


class Criterion:
    """The browser's uninterrupted 25-epoch plateau and 250-epoch rise rule."""

    def __init__(self):
        self.start = None
        self.perfect_start = None
        self.previous = None
        self.plateau = {"start": None, "end": None, "epochs": 0}
        self.perfect = {"start": None, "end": None, "epochs": 0}
        self.last_low = None
        self.first90 = None
        self.rise = None
        self.event = None

    def observe(self, row):
        """Observe exactly one complete-split report per epoch."""
        epoch = row["epoch"]
        if self.previous is not None and epoch != self.previous + 1:
            self.start = self.perfect_start = None
        self.previous = epoch
        if self.first90 is None:
            low = row["heldout_accuracy"] <= .1
            self.start = (self.start if self.start is not None else epoch) if low and row["train_accuracy"] >= .99 else None
            self.perfect_start = (self.perfect_start if self.perfect_start is not None else epoch) if low and row["train_accuracy"] == 1 else None
            for start, name in [(self.start, "plateau"), (self.perfect_start, "perfect")]:
                interval = getattr(self, name)
                if start is not None and (interval["start"] is None or epoch - start > interval["epochs"]):
                    setattr(self, name, {"start": start, "end": epoch, "epochs": epoch - start})
            if low:
                self.last_low = epoch
            if row["heldout_accuracy"] >= .9:
                self.first90 = epoch
                self.rise = None if self.last_low is None else epoch - self.last_low
        if self.event is None and row["heldout_accuracy"] >= .99:
            self.event = {"epoch": epoch, "delayed": self.plateau["epochs"] >= 25 and self.rise is not None and self.rise <= 250}


@torch.no_grad()
def measure(model, data):
    """Evaluate all 9312 rows with one forward and f64 metric reductions."""
    logits = model(data[:, 0], data[:, 1])
    losses = F.cross_entropy(logits, data[:, 2], reduction="none").double()
    correct = (logits.argmax(-1) == data[:, 2]).double()
    split = reference.TRAIN
    values = torch.stack((losses[:split].mean(), correct[:split].mean(), losses[split:].mean(), correct[split:].mean())).cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Non-finite complete-split metrics")
    return dict(zip(("train_loss", "train_accuracy", "heldout_loss", "heldout_accuracy"), values))


def identity_checks(seed):
    """Check the complete task, exact epoch tails and a sampler fingerprint."""
    data = reference.dataset(seed)
    assert data.shape == (9312, 3)
    array = data.numpy()
    assert np.unique(array[:, 0] * 97 + array[:, 1]).size == 9312
    assert np.all((array[:, 1] * array[:, 2]) % 97 == array[:, 0])
    sampler = reference.Sampler(seed)
    batches = [sampler.sample() for _ in range(20)]
    for first in [0, 10]:
        epoch = batches[first:first + 10]
        assert [len(batch) for batch in epoch] == [512] * 9 + [48]
        assert sorted(i for batch in epoch for i in batch) == list(range(4656))
    return {"data_sha256": sha(tensor_bytes(data)), "first_two_epochs_sampler_sha256": sha(np.array([i for batch in batches for i in batch], dtype="<u4").tobytes())}

class Rotate90(LabRotate90):
    """Adapt the real model's input signature to the browser reference."""

    def __init__(self, width, hidden, seed, branch="swiglu"):
        branch_spec = SwiGLU(hidden=hidden) if branch == "swiglu" else DenseBranch(ReLU())
        super().__init__(ModelSpec(width=width, branch=branch_spec), 97, seed)

    def forward(self, a, b):
        return super().forward(torch.stack((a, b), dim=-1))


def validate_initialization(model, config):
    """Preserve E exactly and independently verify the local model adapter."""
    width, seed = config["width"], config["seed"]
    original = reference.Rotate90(width, seed)
    assert torch.equal(model.E, original.E)
    if config["branch"] == "relu":
        assert torch.equal(model.A, original.A)
    data = reference.dataset(seed)[:48]
    native = LabRotate90(model.spec, 97, seed)
    actual = model(data[:, 0], data[:, 1])
    assert torch.equal(actual, native(data[:, :2]))
    if config["branch"] == "relu":
        assert torch.equal(actual, original(data[:, 0], data[:, 1]))
    F.cross_entropy(actual, data[:, 2]).backward()
    F.cross_entropy(native(data[:, :2]), data[:, 2]).backward()
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(model.parameters(), native.parameters()))
    model.zero_grad(set_to_none=True)
    expected = 97 * width + (3 * width * config["hidden"] if config["branch"] == "swiglu" else width * width)
    assert sum(p.numel() for p in model.parameters()) == expected
    return {
        "parameter_count": expected,
        "embedding_sha256": sha(tensor_bytes(model.E)),
        "initial_sha256": sha(b"".join(tensor_bytes(p) for p in model.parameters())),
    }


def summarize(history):
    """Keep first crossings distinct from persistence over a finite budget."""
    criterion = Criterion()
    first = {"train99": None, "test90": None, "test99": None}
    for row in history:
        criterion.observe(row)
        for name, key, target in [("train99", "train_accuracy", .99), ("test90", "heldout_accuracy", .9), ("test99", "heldout_accuracy", .99)]:
            if first[name] is None and row[key] >= target:
                first[name] = row["epoch"]
    after = [] if first["test99"] is None else [r for r in history if r["epoch"] >= first["test99"]]
    dips = [r["epoch"] for r in after if r["heldout_accuracy"] < .99]
    stable = None
    if after and history[-1]["heldout_accuracy"] >= .99:
        stable = dips[-1] + 1 if dips else first["test99"]
    return {
        "first": first, "criterion": vars(criterion),
        "minimum_test_accuracy_after99": min((r["heldout_accuracy"] for r in after), default=None),
        "minimum_train_accuracy_after99": min((r["train_accuracy"] for r in after), default=None),
        "later_epochs_below99": len(dips), "dip_epochs": dips,
        "stable_from_through_budget": stable,
        "final": history[-1],
    }


def run(config, device):
    path = OUTPUT / "runs" / config["id"]
    path.mkdir(parents=True, exist_ok=True)
    result_path = path / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        assert previous["config"] == config
        print(json.dumps({"cached": config["id"], "first99": previous["evaluation"]["first"]["test99"]}), flush=True)
        return previous
    model = Rotate90(config["width"], config["hidden"], config["seed"], config["branch"])
    initialized = validate_initialization(model, config)
    identity = identity_checks(config["seed"])
    model.to(device)
    data = reference.dataset(config["seed"]).to(device)
    sampler = reference.Sampler(config["seed"])
    sf = config["optimizer"] == "schedule_free_adamw"
    settings = dict(lr=config["lr"], betas=(.9, .98), weight_decay=config["weight_decay"], eps=1e-8)
    optimizer = ScheduleFreeAdamW(model.parameters(), **settings, warmup=config["warmup"]) if sf else torch.optim.AdamW(model.parameters(), **settings, foreach=False)
    if sf:
        optimizer.train()
    history, query_history = [], []
    started = time.perf_counter()
    status, error = "budget", None
    try:
        for epoch in range(config["max_epochs"] + 1):
            if epoch:
                model.train()
                for batch_index in range(10):
                    batch = data[sampler.sample()]
                    optimizer.zero_grad(set_to_none=True)
                    F.cross_entropy(model(batch[:, 0], batch[:, 1]), batch[:, 2]).backward()
                    if not sf:
                        step = (epoch - 1) * 10 + batch_index + 1
                        ramp = min(1, step / config["warmup"]) if config["warmup"] else 1
                        optimizer.param_groups[0]["lr"] = config["lr"] * ramp
                    optimizer.step()
            model.eval()
            if sf:
                query_history.append({"epoch": epoch, "step": epoch * 10, **measure(model, data)})
                query = [p.detach().clone() for p in model.parameters()] if epoch <= 1 else None
                optimizer.eval()
            try:
                history.append({"epoch": epoch, "step": epoch * 10, **measure(model, data)})
            finally:
                if sf:
                    optimizer.train()
            if sf and query is not None:
                assert all(torch.equal(a, b) for a, b in zip(query, model.parameters()))
            if epoch and epoch % 250 == 0:
                print(json.dumps({"progress": config["id"], "epoch": epoch, "test": history[-1]["heldout_accuracy"]}), flush=True)
    except FloatingPointError as caught:
        status, error = "diverged", str(caught)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {"config": config, "status": status, "error": error, **initialized, **identity, "evaluation": summarize(history), "query": summarize(query_history) if sf else None, "seconds": elapsed}
    if status != "diverged":
        if sf:
            optimizer.eval()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "mode": "evaluation", "config": config, "epoch": history[-1]["epoch"], "sampler": {"order": sampler.order, "position": sampler.position, "random_state": sampler.random.state}}, path / "checkpoint.pt")
    (path / "history.json").write_text(json.dumps(history) + "\n")
    if sf:
        (path / "query-history.json").write_text(json.dumps(query_history) + "\n")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"done": config["id"], "first99": result["evaluation"]["first"]["test99"], "dips": result["evaluation"]["later_epochs_below99"], "min_test": result["evaluation"]["minimum_test_accuracy_after99"], "plateau": result["evaluation"]["criterion"]["plateau"]["epochs"], "seconds": round(elapsed, 2)}), flush=True)
    return result


def metadata(device):
    package = Path(lean_optim.__file__).resolve().parent
    assert package == LIBRARY / "optimizers/src/lean_optim"
    paths = [Path(__file__).resolve(), Path(str(Path(__file__).resolve()) + ".lock"), REFERENCE, LIBRARY / "python/uv.lock", LIBRARY / "python/pyproject.toml", *sorted(package.glob("*.py")), *[LIBRARY / "python/src/lab" / name for name in ["domain/rotary.py", "infrastructure/nn/rotary.py", "infrastructure/rotary_random.py"]]]
    sources = {str(path): sha(path.read_bytes()) for path in paths}
    for path in paths:
        relative = path.relative_to(LIBRARY) if path.is_relative_to(LIBRARY) else Path("model_reference" if path == REFERENCE else "harness") / path.name
        destination = OUTPUT / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    info = {"created_utc": datetime.datetime.now(datetime.UTC).isoformat(), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "source_sha256": sources, "library_commit": subprocess.check_output(["git", "-C", str(LIBRARY), "rev-parse", "HEAD"], text=True).strip(), "model_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(), "tf32": False, "deterministic": True, "batch": "9x512 + 48 per epoch", "evaluation_batch": 9312, "stop": "Full configured budget; no early stop", "evaluation": "Full train/heldout every epoch, both query and averaged model for ScheduleFree; all summaries are epoch observations", "warmup_units": "optimizer updates; 10 updates = one epoch", "selection": "Pilot on seed 11; full confirmation uses seeds 11,17,23, so seed 11 is not a fresh tuning-independent seed"}
    target = OUTPUT / "metadata.json"
    if target.exists():
        old = json.loads(target.read_text())
        assert old["source_sha256"] == sources
        assert old["torch"] == info["torch"]
    else:
        target.write_text(json.dumps(info, indent=2) + "\n")



def verify(plan, device):
    """Reproduce saved metrics and audit optimizer clocks and evaluation modes."""
    configurations = json.loads(plan.read_text())
    metadata = json.loads((OUTPUT / "metadata.json").read_text())
    for name, digest in metadata["source_sha256"].items():
        relative = Path(name)
        if relative.is_relative_to(LIBRARY) or relative == REFERENCE:
            assert sha(relative.read_bytes()) == digest, name
    modes = 0
    for config in configurations:
        path = OUTPUT / "runs" / config["id"]
        result = json.loads((path / "result.json").read_text())
        history = json.loads((path / "history.json").read_text())
        assert result["config"] == config and result["status"] == "budget"
        assert [row["epoch"] for row in history] == list(range(config["max_epochs"] + 1))
        assert summarize(history) == result["evaluation"]
        checkpoint = torch.load(path / "checkpoint.pt", map_location=device, weights_only=True)
        assert checkpoint["config"] == config and checkpoint["mode"] == "evaluation"
        assert checkpoint["epoch"] == config["max_epochs"]
        assert checkpoint["sampler"]["position"] == reference.TRAIN
        assert sorted(checkpoint["sampler"]["order"]) == list(range(reference.TRAIN))
        model = Rotate90(config["width"], config["hidden"], config["seed"], config["branch"]).to(device)
        model.load_state_dict(checkpoint["model"])
        data = reference.dataset(config["seed"]).to(device)
        expected = {key: value for key, value in history[-1].items() if key not in ("epoch", "step")}
        assert measure(model, data) == expected
        group = checkpoint["optimizer"]["param_groups"][0]
        for name, index in zip(model.state_dict(), group["params"]):
            value, state = checkpoint["model"][name], checkpoint["optimizer"]["state"][index]
            assert torch.isfinite(value).all() and int(state["step"]) == 10 * config["max_epochs"]
            if config["optimizer"] == "schedule_free_adamw":
                assert group["training"] is False and torch.equal(value, state["evaluation"])
                assert all(torch.isfinite(state[key]).all() for key in ("base", "evaluation", "second"))
        modes += 1
        if config["optimizer"] == "schedule_free_adamw":
            query = json.loads((path / "query-history.json").read_text())
            assert [row["epoch"] for row in query] == [row["epoch"] for row in history]
            assert summarize(query) == result["query"]
            optimizer = ScheduleFreeAdamW(model.parameters(), lr=config["lr"], betas=(.9, .98), weight_decay=config["weight_decay"], warmup=config["warmup"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer.train()
            expected = {key: value for key, value in query[-1].items() if key not in ("epoch", "step")}
            assert measure(model, data) == expected
            modes += 1
    print(json.dumps({"checkpoints_verified": len(configurations), "weight_modes_verified": modes, "metrics_exactly_reproduced": True}))

def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "docs/rotate90-swiglu-results/full.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true", help="Verify retained histories and checkpoint metrics without training")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this execution context")
    if args.verify:
        verify(args.plan, device)
        return
    metadata(device)
    results = []
    for config in json.loads(args.plan.read_text()):
        results.append(run(config, device))
        (OUTPUT / (args.plan.stem + "-results.json")).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"complete": args.plan.name, "runs": len(results)}), flush=True)


if __name__ == "__main__":
    main()
