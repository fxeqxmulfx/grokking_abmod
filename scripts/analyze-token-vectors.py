#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3"]
# ///
"""Measure optional Gram, sphere, and Fourier geometry of saved residue vectors."""

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np


def primitive_root(prime):
    if prime == 2:
        return 1
    return next(g for g in range(2, prime) if len({pow(g, k, prime) for k in range(prime - 1)}) == prime - 1)


def concentration(rows, top):
    """Fraction of non-constant Fourier power in the strongest frequencies of an ordered cycle."""
    centered = rows - rows.mean(axis=0, keepdims=True)
    power = (np.abs(np.fft.rfft(centered, axis=0)) ** 2).sum(axis=1)
    # Each frequency below Nyquist also carries its conjugate's equal power.
    power[1:(len(rows) + 1) // 2] *= 2
    power = power[1:]
    if power.sum() == 0:
        return 0.0, []
    order = np.argsort(power)[::-1][:top]
    return float(power[order].sum() / power.sum()), (order + 1).tolist()


def read_vectors(directory):
    metadata = dict(line.split("=", 1) for line in (directory / "run.txt").read_text().splitlines())
    prime = int(metadata["modulus"])
    with (directory / "embeddings.csv").open(newline="") as source:
        vectors = {row["symbol"]: np.array([float(value) for key, value in row.items() if key.startswith("c")])
                   for row in csv.DictReader(source)}
    residues = np.stack([vectors[str(residue)] for residue in range(prime)])
    if not np.isfinite(residues).all():
        raise ValueError(f"nonfinite embeddings: {directory}")
    return metadata, residues


def ordered_cycle(residues, metadata, order):
    if order == "auto":
        order = "discrete-log" if metadata["operation"] == "div" else "residue"
    if order == "discrete-log":
        prime = len(residues)
        root = primitive_root(prime)
        return residues[[pow(root, k, prime) for k in range(prime - 1)]], order
    return residues, order


def gram_geometry(rows):
    """Gram invariants, with explicit handling of zero vectors and collapsed tables."""
    centered = rows - rows.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    spectrum = np.linalg.eigvalsh(gram)
    norms = np.linalg.norm(rows, axis=1)
    valid = norms > 0
    unit = rows[valid] / norms[valid, None]
    cosine = np.clip(unit @ unit.T, -1, 1)
    n = len(unit)
    off = cosine[~np.eye(n, dtype=bool)]
    return {"centered_min_eigenvalue": float(spectrum.min()),
            "centered_max_eigenvalue": float(spectrum.max()),
            "numerical_rank": int(np.linalg.matrix_rank(centered)),
            "zero_vectors": int((~valid).sum()), "norm_min": float(norms.min()),
            "norm_max": float(norms.max()), "norm_std": float(norms.std()),
            "off_diagonal_mean": float(off.mean()) if len(off) else None,
            "off_diagonal_std": float(off.std()) if len(off) else None,
            "orthogonal_max_error": float(np.abs(off).max()) if len(off) else None,
            "simplex_max_error": float(np.abs(off + 1 / (n - 1)).max()) if n > 1 else None}


def sphere_geometry(rows, beta, cap_angle):
    """Finite spherical diagnostics; no dynamical convergence is inferred."""
    norms = np.linalg.norm(rows, axis=1)
    unit = rows[norms > 0] / norms[norms > 0, None]
    n, dimension = unit.shape
    if n == 0:
        return {"unit_vectors": 0, "zero_vectors": len(rows), "packing_bound_log": None}
    cosine = np.clip(unit @ unit.T, -1, 1)
    off = cosine[~np.eye(n, dtype=bool)]
    delta = float(np.sqrt(max(0, 2 - 2 * off.max()))) if len(off) else None
    # Connected components at an angular cutoff are a diagnostic, not spherical caps.
    adjacency = cosine >= np.cos(cap_angle)
    remaining = set(range(n))
    sizes = []
    while remaining:
        frontier = [remaining.pop()]
        size = 0
        while frontier:
            node = frontier.pop()
            size += 1
            neighbors = remaining.intersection(np.flatnonzero(adjacency[node]).tolist())
            remaining.difference_update(neighbors)
            frontier.extend(neighbors)
        sizes.append(size)
    return {"unit_vectors": n, "zero_vectors": int(len(rows) - n),
            "minimum_chord_distance": delta,
            "packing_separation": delta / 2 if delta is not None else None,
            "minimum_angle": float(np.arccos(off.max())) if len(off) else None,
            "angular_diameter": float(np.arccos(off.min())) if len(off) else 0.0,
            "centroid_norm": float(np.linalg.norm(unit.mean(axis=0))),
            "packing_bound_log": float(np.log(2 * dimension) + (dimension - 1) * np.log(8 / delta))
            if delta is not None and delta > 0 else None,
            "interaction_beta": beta,
            "normalized_interaction_energy": float(np.exp(beta * (cosine - 1)).mean() / (2 * beta)),
            "component_angle": cap_angle, "component_sizes": sorted(sizes, reverse=True)}


def fourier_geometry(rows, top):
    """Fit frequency planes; a general learned harmonic is an ellipse or a line."""
    centered = rows - rows.mean(axis=0, keepdims=True)
    coefficients = np.fft.rfft(centered, axis=0)
    power = (np.abs(coefficients) ** 2).sum(axis=1)
    power[1:(len(rows) + 1) // 2] *= 2
    power[0] = 0
    total = power.sum()
    order = np.argsort(power)[::-1][:top] if total > 0 else []
    harmonics = []
    for frequency in order:
        nyquist = len(rows) % 2 == 0 and frequency == len(rows) // 2
        scale = (1 if nyquist else 2) / len(rows)
        a, b = coefficients[frequency].real * scale, -coefficients[frequency].imag * scale
        aa, bb, ab = float(a @ a), float(b @ b), float(a @ b)
        axes = np.sqrt(np.maximum(0, np.linalg.eigvalsh([[aa, ab], [ab, bb]])))
        harmonics.append({"frequency": int(frequency), "power_share": float(power[frequency] / total),
                          "kind": "line" if nyquist else "frequency_plane",
                          "axis_lengths": axes.tolist(),
                          "circle_defect": float(np.sqrt((aa - bb) ** 2 + 4 * ab ** 2) / (aa + bb))
                          if aa + bb > 0 else None})
    retained = np.zeros_like(coefficients)
    for frequency in order:
        retained[frequency] = coefficients[frequency]
    fit = np.fft.irfft(retained, n=len(rows), axis=0)
    energy = float((centered ** 2).sum())
    return {"cycle_length": len(rows), "harmonics": harmonics,
            "relative_squared_fit_error": float(((centered - fit) ** 2).sum() / energy)
            if energy > 0 else 0.0}


def compare_grams(directories):
    comparisons = []
    for i, first in enumerate(directories):
        _, x = read_vectors(first)
        x = x - x.mean(axis=0, keepdims=True)
        for second in directories[i + 1:]:
            _, y = read_vectors(second)
            y = y - y.mean(axis=0, keepdims=True)
            if len(x) != len(y):
                comparisons.append({"first": str(first), "second": str(second),
                                    "skipped": "different residue vocabularies"})
                continue
            gx, gy = x @ x.T, y @ y.T
            denominator = np.linalg.norm(gy)
            comparison = {"first": str(first), "second": str(second),
                          "relative_gram_error": float(np.linalg.norm(gx - gy) / denominator)
                          if denominator > 0 else None}
            if x.shape[1] == y.shape[1] and np.linalg.norm(y) > 0:
                u, _, vt = np.linalg.svd(x.T @ y)
                comparison["orthogonal_alignment_error"] = float(np.linalg.norm(x @ (u @ vt) - y) / np.linalg.norm(y))
            comparisons.append(comparison)
    return comparisons


def analyze(directory, top, permutations, seed, *, gram=False, sphere=False,
            fourier=False, order="auto", beta=1.0, cap_angle=0.2, gram_matrix=False):
    metadata, residues = read_vectors(directory)
    cycle, used_order = ordered_cycle(residues, metadata, order)
    observed, frequencies = concentration(cycle, top)
    generator = np.random.default_rng(seed)
    null = np.array([concentration(cycle[generator.permutation(len(cycle))], top)[0]
                     for _ in range(permutations)])
    centered = residues - residues.mean(axis=0, keepdims=True)
    energy = np.linalg.svd(centered, compute_uv=False) ** 2
    total = energy.sum()
    energy = energy / total if total > 0 else np.zeros_like(energy)
    norms = np.linalg.norm(residues, axis=1, keepdims=True)
    unit = np.divide(residues, norms, out=np.zeros_like(residues), where=norms > 0)
    cosine = unit @ unit.T
    valid = (norms[:, 0] > 0)
    cosine = cosine[np.ix_(valid, valid)]
    np.fill_diagonal(cosine, -np.inf)
    with (directory / "loss.csv").open(newline="") as source:
        last = list(csv.DictReader(source))[-1]
    result = {"run": str(directory.resolve()), "embeddings": metadata["embeddings"],
            "mesh_connectivity": metadata["mesh_connectivity"], "parameters": int(metadata["parameters"]),
            "status": metadata["status"], "stop_reason": metadata.get("stop_reason"),
            "completed_epochs": int(last["completed_epochs"]),
            "train_accuracy": float(last["train_accuracy"]),
            "validation_accuracy": float(last["validation_accuracy"]),
            "embeddings_sha256": hashlib.sha256((directory / "embeddings.csv").read_bytes()).hexdigest(),
            "fourier_order": used_order, "fourier_top_power_fraction": observed, "strongest_frequencies": frequencies,
            "permutation_null_mean": float(null.mean()),
            "permutation_null_p99": float(np.quantile(null, .99)),
            "permutation_p_value": float((1 + (null >= observed).sum()) / (1 + permutations)),
            "effective_rank": float(np.exp(-(energy * np.log(energy, where=energy > 0, out=np.zeros_like(energy))).sum())) if total > 0 else 0.0,
            "directions_for_99_percent_energy": int(np.searchsorted(np.cumsum(energy), .99) + 1) if total > 0 else 0,
            "maximum_pair_cosine": float(cosine.max()) if len(cosine) > 1 else None,
            "mean_nearest_neighbor_cosine": float(cosine.max(axis=1).mean()) if len(cosine) > 1 else None}
    if used_order == "discrete-log":
        result["discrete_log_top_power_fraction"] = observed
    if gram:
        result["gram_geometry"] = gram_geometry(residues)
    if gram_matrix:
        result["gram_matrix"] = (residues @ residues.T).tolist()
    if sphere:
        result["sphere_geometry"] = sphere_geometry(residues, beta, cap_angle)
    if fourier:
        result["fourier_geometry"] = fourier_geometry(cycle, top)
    return result


def self_test():
    """Check known geometries, Parseval weights, degeneracy, and coordinate invariance."""
    orthogonal = np.eye(8)
    assert gram_geometry(orthogonal)["orthogonal_max_error"] == 0
    simplex = (orthogonal - 1 / 8) / np.sqrt(7 / 8)
    assert gram_geometry(simplex)["simplex_max_error"] < 1e-14
    angles = 2 * np.pi * np.arange(17) / 17
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    fit = fourier_geometry(circle, 1)
    assert fit["relative_squared_fit_error"] < 1e-28
    assert fit["harmonics"][0]["circle_defect"] < 1e-14
    sphere = sphere_geometry(circle, 1.0, 0.2)
    assert abs(sphere["minimum_chord_distance"] - 2 * np.sin(np.pi / 17)) < 1e-14
    assert sphere["component_sizes"] == [1] * 17
    ellipse = circle * [2, 1]
    assert abs(fourier_geometry(ellipse, 1)["harmonics"][0]["circle_defect"] - .6) < 1e-14
    k = np.arange(16)
    mixed = np.column_stack([np.cos(2 * np.pi * k / 16), (-1.0) ** k])
    share, frequencies = concentration(mixed, 1)
    assert frequencies == [8] and abs(share - 2 / 3) < 1e-14
    assert fourier_geometry(mixed, 1)["harmonics"][0]["kind"] == "line"
    zeros = np.zeros((3, 4))
    assert concentration(zeros, 2) == (0.0, [])
    assert fourier_geometry(zeros, 2)["relative_squared_fit_error"] == 0
    assert sphere_geometry(zeros, 1, .2)["unit_vectors"] == 0
    assert primitive_root(2) == 1
    assert ordered_cycle(circle, {"operation": "add"}, "auto")[1] == "residue"
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    with tempfile.TemporaryDirectory(prefix="embedding-geometry-") as temporary:
        directories = []
        for name, rows in [("original", circle), ("rotated", circle @ rotation), ("zero", np.zeros_like(circle))]:
            directory = Path(temporary) / name
            directory.mkdir()
            (directory / "run.txt").write_text("modulus=17\noperation=add\nembeddings=table\nmesh_connectivity=neighbor\nparameters=34\nstatus=complete\n")
            with (directory / "embeddings.csv").open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["symbol", "c0", "c1"])
                writer.writerows([[i, *row] for i, row in enumerate(rows)])
            (directory / "loss.csv").write_text("completed_epochs,train_accuracy,validation_accuracy\n1,1,1\n")
            report = analyze(directory, 1, 4, 7, gram=True, sphere=True, fourier=True)
            json.dumps(report, allow_nan=False)
            if name == "zero":
                assert report["effective_rank"] == 0 and report["maximum_pair_cosine"] is None
            directories.append(directory)
        comparison = compare_grams(directories[:2])[0]
        assert comparison["relative_gram_error"] < 1e-14
        assert comparison["orthogonal_alignment_error"] < 1e-14
    print("PASS embedding geometry: Gram, simplex, circle/ellipse, packing, Fourier/Nyquist, zero vectors, orthogonal alignment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path)
    parser.add_argument("--self-test", action="store_true", help="Check analytic examples and degenerate tables")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gram", action="store_true", help="Gram spectrum, rank, equiangular/simplex defects")
    parser.add_argument("--gram-matrix", action="store_true", help="Include the full raw Gram matrix in JSON")
    parser.add_argument("--compare-gram", action="store_true", help="Compare all run pairs, including orthogonal alignment")
    parser.add_argument("--sphere", action="store_true", help="Separation, packing bound, energy, angular components")
    parser.add_argument("--fourier", action="store_true", help="Fit frequency planes and measure circle defects")
    parser.add_argument("--fourier-order", choices=["auto", "residue", "discrete-log"], default="auto")
    parser.add_argument("--energy-beta", type=float, default=1.0)
    parser.add_argument("--cap-angle", type=float, default=0.2, help="Angular component threshold in radians")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return
    if not arguments.directories:
        parser.error("provide at least one run directory")
    if arguments.top <= 0 or arguments.permutations <= 0:
        parser.error("top and permutations must be positive")
    if not np.isfinite(arguments.energy_beta) or arguments.energy_beta <= 0:
        parser.error("energy-beta must be finite and positive")
    if not np.isfinite(arguments.cap_angle) or not 0 < arguments.cap_angle <= np.pi:
        parser.error("cap-angle must be in (0, pi]")
    report = {"protocol": {"order": arguments.fourier_order, "top_frequencies": arguments.top,
                           "null": "random_cycle_order", "permutations": arguments.permutations,
                           "seed": arguments.seed, "cosine": "between_distinct_residue_vectors"},
              "runs": [analyze(directory, arguments.top, arguments.permutations, arguments.seed,
                               gram=arguments.gram, sphere=arguments.sphere, fourier=arguments.fourier,
                               order=arguments.fourier_order, beta=arguments.energy_beta,
                               cap_angle=arguments.cap_angle, gram_matrix=arguments.gram_matrix)
                       for directory in arguments.directories]}
    if arguments.compare_gram:
        report["gram_comparisons"] = compare_grams(arguments.directories)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)
        print(arguments.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
