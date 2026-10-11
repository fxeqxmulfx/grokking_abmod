# Embedding geometry specifications

This is an independent Lean 4 project pinned to Lean/Mathlib **4.34.1**. It
depends on Mathlib; it does not depend on the sibling Transformer project.

```bash
cd lean
lake exe cache get
lake build
lake env lean Audit.lean
```

`Audit.lean` checks every theorem in the `Grokking` namespace transitively,
accepting only `propext`, `Classical.choice`, and `Quot.sound`. There are no
`sorry` declarations or additional axioms.

| Module | Runtime option / mathematical contract |
| --- | --- |
| `Gram` | `--gram`, `--compare-gram`: PSD Gram representations and orthogonal equivalence of configurations with equal Gram matrices |
| `Shapes` | `--embedding-form circle/fourier/orthogonal/simplex/equiangular/antipodal`: circle and frequency-plane identities, equiangular Gram characterization and both overlap bounds |
| `Simplex` | Canonical regular simplex in the zero-sum hyperplane, with unit norms and overlap `-1/(V-1)` |
| `Codewords` | Regular angular code separation `2*pi/M` and covering radius `pi/M`; applies to the fundamental frequency, before any learned mesh |
| `BitCodes` | Injective leading bit codes and preservation of injectivity when additional parity features are appended; wide codes only |
| `SignedCodes` | The exact 256 signed-permutation codes used by the Rust option: distinct codes, integer coordinate cap four, squared norm thirty |
| `Basic` | `--embedding-norm unit/rms`: radius above the positive normalization floor; corrected RMSNorm identity with positive epsilon |
| `Residual` | `--preserve-features N`: code injection, kernel conditions on final output maps, preservation through arbitrary nonlinear residual interiors, and distinction of raw hidden states |
| `RoPE` | `--rope 1`: split-half norm/distance preservation and the relative-position identity |
| `XSA` | `--xsa 1`: orthogonality to the value direction/value above the positive normalization floor |
| `Packing` | `--sphere`: upper packing bound for separated finite unit configurations and lower bound for maximal separated configurations |

Theorems specify real arithmetic. They do not prove Rust execution, CUDA memory
safety, compiler correctness, or rounding-error bounds. Rust validates geometry
dimensions and uses numerical tests for its Helmert simplex construction. The
CPU reference independently differentiates in f64; CUDA verification compares
effective embeddings, predictions, derivatives, and optimizer updates in f32.

In particular, the equiangular statements describe a configuration satisfying
the specified Gram condition. They do not prove that unconstrained training
stays equiangular, that a learned embedding mesh realizes arbitrary tables, or
that an orthogonal transformation preserves predictions of a nonlinear mesh.
Protected residual features refer to states **before final RMSNorm** and require
distinct input code features for the tokens one wants to distinguish.

Ports are attributed to `lean_projects/transformer`, revision
`033597a7eb91960042a50934e11f07860a213f8d`:

- `Transformer.GPTMini.Sparsemax.EmbeddingGram` and `Transformer.Homogenized.HeadConj`;
- `Transformer.Metastability.AngularEnergy`, `Transformer.Homogenized.Simplex`, and `Transformer.Kinetic.Codewords`;
- `Transformer.GPTMini.Sparsemax.PairedRecallCodes`;
- `Transformer.GPTMini.Bridge.SphereResidence` and `Transformer.GPTMini.Semantics.ResidualFeature`;
- `Transformer.GPTMini.RoPE` and `Transformer.Causal.Packing.{Basic,Upper,Lower,Count}`.

Namespace/import changes make these ports standalone. Residual interiors have
been generalized to arbitrary functions so the statement covers the project's
ELU meshes. Epsilon-free normalization has been replaced with the positive
epsilon/floor actually used here. Frequency-plane identities, bit-code proofs,
and the upper equiangular-overlap bound are additions for this project.
