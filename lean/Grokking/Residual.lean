import Grokking.Basic

/-!
Adapted from Transformer.GPTMini.Semantics.ResidualFeature at
033597a7eb91960042a50934e11f07860a213f8d. Attention and FFN interiors are
arbitrary functions, allowing the actual ELU meshes, QKNorm, RoPE, and XSA.
Only their final linear output maps must have image in the probe's kernel.
-/

namespace Grokking

noncomputable def residualBlock {d : ℕ} (attention feedForward : EucSpace d → EucSpace d)
    (outputA outputF : EucSpace d →ₗ[ℝ] EucSpace d) (x : EucSpace d) : EucSpace d :=
  let y := x + outputA (attention x)
  y + outputF (feedForward y)

theorem block_probe_preserved {d : ℕ} (probe : EucSpace d →ₗ[ℝ] ℝ)
    (attention feedForward : EucSpace d → EucSpace d)
    (outputA outputF : EucSpace d →ₗ[ℝ] EucSpace d)
    (ha : probe.comp outputA = 0) (hf : probe.comp outputF = 0) (x : EucSpace d) :
    probe (residualBlock attention feedForward outputA outputF x) = probe x := by
  have hA (z : EucSpace d) : probe (outputA z) = 0 := by
    have h := congrArg (fun f : EucSpace d →ₗ[ℝ] ℝ => f z) ha
    simpa only [LinearMap.comp_apply, LinearMap.zero_apply] using h
  have hF (z : EucSpace d) : probe (outputF z) = 0 := by
    have h := congrArg (fun f : EucSpace d →ₗ[ℝ] ℝ => f z) hf
    simpa only [LinearMap.comp_apply, LinearMap.zero_apply] using h
  simp [residualBlock, probe.map_add, hA, hF]

noncomputable def hidden {d : ℕ} (blocks : ℕ → EucSpace d → EucSpace d)
    (input : EucSpace d) : ℕ → EucSpace d
  | 0 => input
  | n + 1 => blocks n (hidden blocks input n)

theorem hidden_probe_preserved {d : ℕ} (probe : EucSpace d →ₗ[ℝ] ℝ)
    (blocks : ℕ → EucSpace d → EucSpace d)
    (hb : ∀ n x, probe (blocks n x) = probe x) (input : EucSpace d) (n : ℕ) :
    probe (hidden blocks input n) = probe input := by
  induction n with
  | zero => rfl
  | succ n ih => rw [hidden, hb, ih]

theorem hidden_probe_distinct {d : ℕ} (probe : EucSpace d →ₗ[ℝ] ℝ)
    (blocks : ℕ → EucSpace d → EucSpace d)
    (hb : ∀ n x, probe (blocks n x) = probe x) (x y : EucSpace d)
    (hxy : probe x ≠ probe y) (n : ℕ) : hidden blocks x n ≠ hidden blocks y n := by
  intro heq
  apply hxy
  have h := congrArg probe heq
  simpa only [hidden_probe_preserved probe blocks hb] using h

/-- Copying the code at each forward pass fixes selected input coordinates even
while all unconstrained embedding-mesh parameters continue to train. -/
noncomputable def injectCode {d : ℕ} (count : ℕ) (code x : EucSpace d) : EucSpace d :=
  WithLp.toLp 2 fun i => if i.val < count then code i else x i

theorem injectCode_preserved {d : ℕ} (count : ℕ) (code x : EucSpace d)
    (i : Fin d) (hi : i.val < count) : injectCode count code x i = code i := by
  simp [injectCode, hi]

/-- Each two-parent output row vanishes when both of its weights are constrained. -/
theorem mesh_output_zero (a b x y : ℝ) (ha : a = 0) (hb : b = 0) : a * x + b * y = 0 := by
  rw [ha, hb]; ring

example (probe : EucSpace 2 →ₗ[ℝ] ℝ) : probe.comp (0 : EucSpace 2 →ₗ[ℝ] EucSpace 2) = 0 := by simp
example : (0 : ℝ) = 0 ∧ (0 : ℝ) = 0 := ⟨rfl, rfl⟩

end Grokking
