# Optional embedding geometry

Geometric embeddings and diagnostics are available behind runtime flags in
`mesh-gpt train` and `mesh-gpt grok`. Defaults retain the original mesh-generated
parity codes, RMSNorm/QK-Norm, RoPE, XSA, and AdamW behavior. The corresponding
real-valued proofs live in the standalone [Lean project](../lean/README.md).

## Token-vector forms

`--embedding mesh|table` chooses the parameterization. With `mesh`, a form
specifies fixed input codes to an ELU mesh; its learned output need not retain
their norms or angles. With `table`, non-default forms initialize the actual
tied vectors directly. The default `parity` table ablation continues to tabulate
the seeded mesh, preserving its initial function.

| `--embedding-form` | Codes / geometric initialization | Required width |
| --- | --- | --- |
| `parity` | Original signed bit and parity codes; signed radix packing at narrow widths | At least 2 |
| `circle` | One unit `(cos, sin)` pair | At least 2 |
| `fourier` | Consecutive harmonic pairs with equal weights and unit total norm | At least twice the harmonic count |
| `orthogonal` | Standard basis vectors | At least vocabulary size `V` |
| `simplex` | Unit regular simplex, overlap `-1/(V-1)`, using Helmert coordinates | At least `V-1` |
| `equiangular` | Unit vectors with requested common overlap | `V` in the interior; `V-1` at the simplex boundary; 1 at overlap 1 |
| `antipodal` | Alternating `+u/-u`; a deliberate two-cluster control | At least 2 |
| `hypercube` | Unit signed binary-digit codes | At least `ceil(log2 V)` |
| `signed-permutations` | 256 distinct signed permutations of `(1,2,3,4)`, squared norm 30 | At least 4, `V<=256` |
| `sphere` | Deterministic normalized Gaussian directions | At least 2 |

All configurations also satisfy the model's head-width and capacity limits.
`--equiangular-overlap X` defaults to zero and must lie in `[-1/(V-1),1]`.
An interior equiangular Gram has rank `V`; the simplex boundary has rank `V-1`.
Consequently an interior configuration cannot be represented in width `V-1`.

`--embedding-frequency N` selects the first cyclic frequency (default 1).
`--embedding-harmonics N` selects the number of consecutive Fourier pairs
(default 1). Frequencies are reduced modulo the cycle length. A non-coprime
single frequency can identify different tokens; regular-code separation applies
to the fundamental frequency or a coprime permutation of it. An antipodal form
also identifies tokens when the vocabulary contains more than two identities.

`--geometry-seed N` controls sphere codes. It is independent of weight
initialization except when arithmetic ordering uses the grok dataset seed.

## Arithmetic ordering

`--embedding-order token` is the default. `residue` orders the grok codes by
residue, and `discrete-log` orders nonzero residues by powers of a deterministic
primitive root. Division subtracts discrete logarithms; addition/subtraction
use the residue cycle. The original seeded token permutation is reconstructed
from the stored modulus and seed, including after checkpoint loading.

Arithmetic order is available only for `grok`. It explicitly supplies an
arithmetic prior and is recorded as `geometry_supplies_arithmetic_order=true`.
The dataset's token IDs still use the same seeded permutation.

Circle/Fourier arithmetic codes reserve independent coordinates for the
operation and `=`; discrete-log codes additionally reserve a coordinate for
zero. The required width is therefore `2*harmonics+2` for residue order or
`2*harmonics+3` for discrete-log order. `circle` always has one harmonic.

```bash
# Learned table starting on a regular residue circle.
cargo oxide run --bin mesh-gpt -- grok --operation add --modulus 97 \
  --embedding table --embedding-form circle --embedding-order residue \
  --output target/geometry-circle

# A fixed multi-frequency prior for division.
cargo oxide run --bin mesh-gpt -- grok --operation div --modulus 97 \
  --embedding table --embedding-form fourier --embedding-harmonics 8 \
  --embedding-order discrete-log --freeze-embedding \
  --output target/geometry-fourier
```

## Constraints during training

`--freeze-embedding` excludes the embedding group from AdamW, including weight
decay. A geometric **table** therefore retains its initialized geometry. Other
parameters continue to train. Diagnostic optimizer gradients for the frozen
group are zero in both implementations.

`--embedding-norm none|unit|rms` acts on the effective tied vectors at every
forward pass, including both input lookup and unembedding. `unit` has radius
one and `rms` radius `sqrt(width)` when the raw norm is at least `1e-12`.
Below this floor the mapping is linear; zero remains zero. The CPU reference
and CUDA backward differentiate this mapping, combining input and head
contributions before differentiating the embedding mesh once.

`--preserve-features N` copies the first N code coordinates into the effective
tied vectors, independently of their learned raw values. It constrains the
corresponding final rows of every attention-output and FFN mesh to zero. Those
weights and optimizer gradients remain zero throughout training. Parameter
setters and checkpoint readers reject violations.

Each protected coordinate is preserved through all residual blocks. Distinct
protected code features force distinct raw hidden states. For default wide
parity codes, `ceil(log2 V)` leading coordinates distinguish the whole
vocabulary. This does not assert that the final normalized readout predicts the
correct answer or that the remaining learned coordinates retain high rank.

Full-vector normalization would rescale protected coordinates, so combining
`--preserve-features N` with `--embedding-norm unit|rms` is rejected. Select one
constraint, or use a frozen geometric table.

```bash
# Preserve all token identities while training the remaining mesh coordinates.
cargo oxide run --bin mesh-gpt -- grok --modulus 97 --preserve-features 7 \
  --output target/geometry-protected

# Normalize the generated embeddings during learning.
cargo oxide run --bin mesh-gpt -- grok --embedding-norm unit \
  --output target/geometry-unit
```

`--rope 0|1`, `--qk-norm 0|1`, and `--xsa 0|1` independently select existing
operators (all default to 1). Turning off RoPE permits odd head widths, with
at least two channels per head. The final and pre-block RMSNorms remain present.

## Optional analysis

The existing script reads the saved **effective** vectors, including the
normalization or protected-coordinate mapping. Each diagnostic has a flag:

```bash
uv run --locked scripts/analyze-token-vectors.py \
  target/geometry-circle target/geometry-unit \
  --gram --compare-gram --sphere --fourier \
  --output target/embedding-geometry.json
```

- `--gram`: centered Gram spectrum/rank, norms, normalized off-diagonal overlaps,
  and orthogonal/simplex defects. `--gram-matrix` includes the full raw Gram.
- `--compare-gram`: centered Gram differences and, at equal widths, optimal
  orthogonal alignment errors between runs on the same residue vocabulary.
- `--sphere`: distances and angles between normalized nonzero vectors,
  centroid norm, angular connected components, and normalized interaction energy.
  The packing upper bound uses half the measured minimum chord distance, so
  the separation is strict as in the Lean theorem. The lower packing theorem
  requires maximality and is not applied to arbitrary learned tables.
- `--fourier`: selected frequency planes, ellipse axis lengths, circle defects,
  and reconstruction error. A Nyquist component is reported as a line. Both
  conjugate frequencies contribute to real Fourier power.

`--fourier-order auto` uses discrete logarithms for division and residue order
for other operations. Explicit `residue` and `discrete-log` overrides are also
available. `--energy-beta` and `--cap-angle` select the sphere diagnostic's
positive inverse temperature and angular component cutoff in radians.

Zero vectors and constant tables are reported explicitly; they are not treated
as unit directions. Gram comparisons are statements about geometry, and do not
assert equivalent predictions of nonlinear mesh models. Angular components
are measured configurations, with no theorem about training convergence.

## Checkpoints and verification

`MSHGPT09` stores the complete geometry, arithmetic permutation seed, frozen
embedding policy, protected-feature count, and operator switches. Default
models still write the exact `MSHGPT07` format, and previous mesh/table and
connectivity ablations still use `MSHGPT08`. All three formats load.
Grok metadata records every option; text runs also save `geometry.txt`.

```bash
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo oxide run --bin mesh-gpt -- verify
uv run --locked scripts/analyze-token-vectors.py --self-test
cd lean
lake build
lake env lean Audit.lean
```

The CUDA suite checks all forms with both parameterizations, normalization
derivatives, protected residual states, frozen AdamW updates, arithmetic
ordering, and operator switches against independent f64 autodiff. It also
retains the original attention, RMSNorm, RoPE, XSA, and reduction checks.

Continuous spherical clustering/metastability theorems and convex Gram or
local-sparsemax optimization are not imported as guarantees for this model:
they require different architectures or dynamics. See the
[Lean contracts](../lean/README.md) for exact hypotheses and attribution.
