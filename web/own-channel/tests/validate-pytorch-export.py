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
"""Validate actual exported PyTorch modules against independent browser fixtures."""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

torch.set_num_threads(2)
directory = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/grokking-pytorch-fixtures")


def floats(path: Path, dtype: torch.dtype) -> torch.Tensor:
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype)


def flat(model, gradients=False):
    return torch.cat([(parameter.grad if gradients else parameter).detach().flatten()
                      for parameter in model.parameters()])


for location in sorted(directory.iterdir()):
    if not location.is_dir():
        continue
    reference = json.loads((location / "reference.json").read_text())
    spec = importlib.util.spec_from_file_location(location.name, location / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (module.DEFAULT_WIDTH, module.DEFAULT_SEED, module.DEFAULT_EPOCHS) == (reference["width"], 11, 17)
    assert (module.DEFAULT_LEARNING_RATE, module.DEFAULT_WEIGHT_DECAY) == (reference["learningRate"], reference["weightDecay"])
    model = module.Rotate90()
    assert list(dict(model.named_parameters())) == ["E", "A"]
    assert sum(parameter.numel() for parameter in model.parameters()) == reference["parameters"]
    torch.testing.assert_close(flat(model), floats(location / "initial.f32", torch.float32), rtol=0, atol=0)
    expected_data = torch.tensor(list(struct.iter_unpack("<III", (location / "data.u32").read_bytes())))
    rows = module.dataset(11)
    assert torch.equal(rows, expected_data)
    sampler = module.Sampler(11)
    assert [sampler.sample() for _ in reference["batches"]] == reference["batches"]
    model.double()
    examples = rows[reference["ids"]]
    logits = model(examples[:, 0], examples[:, 1]).detach().flatten()
    torch.testing.assert_close(logits, floats(location / "logits.f64", torch.float64), rtol=0, atol=2e-12)
    batch = rows[reference["sample"]]
    loss = F.cross_entropy(model(batch[:, 0], batch[:, 1]), batch[:, 2])
    assert abs(loss.item() - reference["loss"]) < 2e-12
    loss.backward()
    torch.testing.assert_close(flat(model, True), floats(location / "gradient.f64", torch.float64), rtol=0, atol=2e-12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=module.DEFAULT_LEARNING_RATE, betas=(0.9, 0.98),
                                 eps=1e-8, weight_decay=module.DEFAULT_WEIGHT_DECAY, foreach=False)
    for step in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(batch[:, 0], batch[:, 1]), batch[:, 2]).backward()
        optimizer.param_groups[0]["lr"] = module.DEFAULT_LEARNING_RATE * min(step / 10, 1)
        optimizer.step()
    torch.testing.assert_close(flat(model), floats(location / "adam.f64", torch.float64), rtol=0, atol=2e-12)
    print(f"PASS {location.name}: exact initialization/data/21 batches; logits, all gradients and three AdamW updates match f64.")
