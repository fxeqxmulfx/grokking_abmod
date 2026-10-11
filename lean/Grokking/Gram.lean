import Grokking.Basic

/-!
Adapted from Transformer.GPTMini.Sparsemax.EmbeddingGram and
Transformer.Homogenized.HeadConj at 033597a7eb91960042a50934e11f07860a213f8d.
Only geometry of the effective table is ported; convex optimization of an
unconstrained Gram is a different architecture and is not assumed here.
-/

open scoped BigOperators

namespace Grokking

def featureGram {ι : Type*} {D : ℕ} (features : Fin D → ι → ℝ) : Matrix ι ι ℝ :=
  ∑ d, Matrix.vecMulVec (features d) (features d)

theorem featureGram_apply {ι : Type*} {D : ℕ} (features : Fin D → ι → ℝ) (i j : ι) :
    featureGram features i j = ∑ d, features d i * features d j := by
  rw [featureGram, Matrix.sum_apply]
  exact Finset.sum_congr rfl fun d _ => Matrix.vecMulVec_apply _ _ _ _

/-- The feature dimension is existential, so the converse imposes no small rank. -/
theorem posSemidef_iff_featureGram {ι : Type*} [Finite ι] (G : Matrix ι ι ℝ) :
    G.PosSemidef ↔ ∃ D : ℕ, ∃ features : Fin D → ι → ℝ, G = featureGram features := by
  rw [Matrix.posSemidef_iff_eq_sum_vecMulVec]
  constructor
  · rintro ⟨D, features, h⟩
    exact ⟨D, features, by simpa only [featureGram, Pi.star_def, star_trivial] using h⟩
  · rintro ⟨D, features, h⟩
    exact ⟨D, features, by simpa only [featureGram, Pi.star_def, star_trivial] using h⟩

theorem featureGram_posSemidef {ι : Type*} [Finite ι] {D : ℕ}
    (features : Fin D → ι → ℝ) : (featureGram features).PosSemidef :=
  (posSemidef_iff_featureGram _).2 ⟨D, features, rfl⟩

theorem norm_linearCombination_eq {d n : ℕ} {X Y : Fin n → EucSpace d}
    (hgram : ∀ k l, inner (𝕜 := ℝ) (X k) (X l) = inner (𝕜 := ℝ) (Y k) (Y l))
    (c : Fin n → ℝ) :
    ‖Fintype.linearCombination ℝ X c‖ = ‖Fintype.linearCombination ℝ Y c‖ := by
  have h : ∀ Z : Fin n → EucSpace d, ‖Fintype.linearCombination ℝ Z c‖ ^ 2 =
      ∑ k, ∑ l, c k * c l * inner (𝕜 := ℝ) (Z k) (Z l) := fun Z => by
    rw [← real_inner_self_eq_norm_sq, Fintype.linearCombination_apply, sum_inner]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [inner_sum]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [inner_smul_left, inner_smul_right]
    simp only [conj_trivial]; ring
  rw [← sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _), h, h]
  simp only [hgram]

/-- Equal Gram matrices determine shape up to orthogonal equivalence. This is
a statement about configurations, not a symmetry of the nonlinear mesh model. -/
theorem exists_linearIsometryEquiv_of_inner_eq {d n : ℕ} {X Y : Fin n → EucSpace d}
    (hgram : ∀ k l, inner (𝕜 := ℝ) (X k) (X l) = inner (𝕜 := ℝ) (Y k) (Y l)) :
    ∃ O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, ∀ k, O (X k) = Y k := by
  set fX := Fintype.linearCombination ℝ X
  set fY := Fintype.linearCombination ℝ Y
  have hker : LinearMap.ker fX ≤ LinearMap.ker fY := fun c hc => by
    rw [LinearMap.mem_ker, ← norm_eq_zero] at hc ⊢
    rw [← norm_linearCombination_eq hgram c]; exact hc
  set L₀ : LinearMap.range fX →ₗ[ℝ] EucSpace d :=
    (LinearMap.ker fX).liftQ fY hker ∘ₗ fX.quotKerEquivRange.symm.toLinearMap
  have hL₀ : ∀ c, L₀ ⟨fX c, LinearMap.mem_range_self fX c⟩ = fY c := fun c => by simp [L₀]
  let L : LinearMap.range fX →ₗᵢ[ℝ] EucSpace d :=
    { toLinearMap := L₀
      norm_map' := by
        rintro ⟨z, c, rfl⟩
        exact (hL₀ c).symm ▸ (norm_linearCombination_eq hgram c).symm }
  refine ⟨L.extend.toLinearIsometryEquiv rfl, fun k => ?_⟩
  have hk : fX (Pi.single k 1) = X k := by simp [fX, Fintype.linearCombination_apply_single]
  have := LinearIsometry.extend_apply L ⟨fX (Pi.single k 1), LinearMap.mem_range_self fX _⟩
  simp only [L, LinearIsometry.coe_mk, hL₀] at this
  simp only [LinearIsometry.coe_toLinearIsometryEquiv]
  rw [← hk, this]
  simp [fY, Fintype.linearCombination_apply_single]

example : (0 : Matrix (Fin 2) (Fin 2) ℝ).PosSemidef := Matrix.PosSemidef.zero
example : ∀ _ _ : Fin 2, inner (𝕜 := ℝ) (0 : EucSpace 2) 0 = inner (𝕜 := ℝ) (0 : EucSpace 2) 0 :=
  fun _ _ => rfl

end Grokking
