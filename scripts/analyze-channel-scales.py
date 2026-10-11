#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Hidden-channel scales of one-hot one-layer ReLU MLPs and own-channel mesh MLPs.

Hidden channel c adds R_c g_c relu(E_c x) to the logits, with input column E_c,
readout row R_c and, in an own-channel mesh, the product g_c of its rectified
mesh gains. Its scale is m_c = |E_c| g_c |R_c|, read from the final weights.
The softmax ignores the part of R_c that is equal for every class, so the
centered scale replaces R_c by R_c minus its mean.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

TOP = 16


def weights(directory):
    """Run settings, mesh layers, stored mesh layers and every hidden channel's weights."""
    settings = dict(line.split("=", 1) for line in (directory / "run.txt").read_text().splitlines())
    if settings["status"] != "complete" or settings["encoding"] != "one-hot":
        raise ValueError(f"Expected a complete one-hot run: {directory}")
    width, classes = int(settings["width"]), int(settings["output_classes"])
    if settings["architecture"] == "mlp" and settings["depth"] == "1":
        layers = stored = 0
    elif settings["architecture"] == "mesh-mlp" and settings["connectivity"] == "own":
        layers = int(settings["mesh_layers"])
        # Runs from before tied meshes have no tied_mesh entry and store every layer.
        stored = min(layers, 1) if settings.get("tied_mesh") == "true" else layers
    else:
        raise ValueError(f"Expected a one-layer MLP or an own-channel mesh MLP: {directory}")
    data = (directory / "model.bin").read_bytes()
    readout, mesh = 2 * classes * width, 3 * classes * width
    count = mesh + stored * width
    if len(data) != 4 * count or count != int(settings["parameters"]):
        raise ValueError(f"Unexpected weight count: {directory}")
    values = struct.unpack(f"<{count}f", data)
    channels = [{"a": [values[row * width + c] for row in range(classes)],
                 "b": [values[(classes + row) * width + c] for row in range(classes)],
                 "readout": values[readout + c * classes:readout + (c + 1) * classes],
                 "gains": [values[mesh + layer * width + c] for layer in range(stored)]}
                for c in range(width)]
    return settings, layers, stored, channels, hashlib.sha256(data).hexdigest()


def norm(vector):
    return math.sqrt(sum(value ** 2 for value in vector))


def concentration(scales):
    """Shares of the summed scales in the largest channels, and the channel count for 90%."""
    scales = sorted(scales, reverse=True)
    total, cumulative = sum(scales), 0.0
    for needed, scale in enumerate(scales, 1):
        cumulative += scale
        if cumulative >= .9 * total:
            break
    return {"largest_share": scales[0] / total, "top_share": sum(scales[:TOP]) / total,
            "channels_for_90_percent": needed,
            "effective_channels": total ** 2 / sum(scale ** 2 for scale in scales)}


def analyze(directory):
    settings, layers, stored, channels, digest = weights(directory)
    full, centered, always = [], [], []
    for channel in channels:
        # ReLU passes only positive gains, and a tied gain acts in every mesh layer.
        product = math.prod(max(gain, 0.0) for gain in channel["gains"])
        gain = product ** (layers // stored) if stored else 1.0
        size = norm(channel["a"] + channel["b"]) * gain
        mean = sum(channel["readout"]) / len(channel["readout"])
        full.append(size * norm(channel["readout"]))
        centered.append(size * norm([value - mean for value in channel["readout"]]))
        # With one-hot operands the pre-activation of pair (a, b) is E_c[a] + E_c[b].
        if min(channel["a"]) + min(channel["b"]) > 0:
            always.append({"scale_share": full[-1], "readout_mean_share":
                           len(channel["readout"]) * mean ** 2 / norm(channel["readout"]) ** 2})
    for channel in always:
        channel["scale_share"] /= sum(full)
    spread = None
    if stored > 1:
        # Untied gains of one channel start equal and receive equal updates.
        spread = max((max(channel["gains"]) - min(channel["gains"])) / largest
                     for channel in channels
                     if (largest := max(abs(gain) for gain in channel["gains"])) > 0)
    return {"run": str(directory), "family": settings["family"], "parameters": int(settings["parameters"]),
            "epochs": int(settings["completed_epochs"]), "mesh_layers": layers, "stored_mesh_layers": stored,
            "model_sha256": digest, "channels": len(channels),
            "largest_relative_gain_spread": spread,
            "scales": concentration(full), "centered_scales": concentration(centered),
            "always_active_channels": len(always),
            "always_active_scale_share": sum(channel["scale_share"] for channel in always),
            "always_active": sorted(always, key=lambda channel: -channel["scale_share"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = {"definition": {
        "scale": "|E_c| prod_l relu(u_lc) |R_c| of the final weights",
        "centered_scale": "the scale with R_c minus its mean over the classes",
        "top_share": f"share of the summed scales in the {TOP} largest channels",
        "effective_channels": "(sum m_c)^2 / sum m_c^2",
        "always_active": "channels with E_c[a] + E_c[b] > 0 for every operand pair",
        "readout_mean_share": "classes * mean(R_c)^2 / |R_c|^2",
        "largest_relative_gain_spread": "max over channels of (max_l u_lc - min_l u_lc) / max_l |u_lc|"},
        "runs": [analyze(directory) for directory in arguments.directories]}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)
        print(arguments.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
