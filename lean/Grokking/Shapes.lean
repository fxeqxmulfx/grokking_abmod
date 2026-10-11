import Grokking.Basic

/-!
Circle geometry is adapted from Transformer.Metastability.AngularEnergy.
The equiangular predicate comes from Transformer.Homogenized.Simplex.
The frequency-plane identities and corrected overlap bounds are new ports for
the modular arithmetic experiments; no attention-dynamics convergence is assumed.
-/

open scoped BigOperators

namespace Grokking

noncomputable def circle (θ : ℝ) : EucSpace 2 :=
  (EuclideanSpace.equiv (Fin 2) ℝ).symm ![Real.cos θ, Real.sin θ]

theorem circle_norm (θ : ℝ) : ‖circle θ‖ = 1 := by
  rw [EuclideanSpace.norm_eq]
  simp only [circle, Fin.sum_univ_two, Real.norm_eq_abs, sq_abs]
  norm_num [Real.sin_sq_add_cos_sq, Real.cos_sq_add_sin_sq]

theorem circle_inner (θ φ : ℝ) :
    inner (𝕜 := ℝ) (circle θ) (circle φ) = Real.cos (θ - φ) := by
  simp only [circle, PiLp.inner_apply, RCLike.inner_apply, conj_trivial,
    Fin.sum_univ_two, Real.cos_sub]
  simp [Matrix.cons_val_zero, Matrix.cons_val_one]
  ring

/-- Direct sum of weighted harmonic pairs; unused model coordinates may be zero-padded. -/
noncomputable def fourierCode {m : ℕ} (weights frequencies : Fin m → ℝ) (θ : ℝ) :
    EuclideanSpace ℝ (Fin m × Fin 2) :=
  WithLp.toLp 2 fun p => weights p.1 *
    if p.2 = 0 then Real.cos (frequencies p.1 * θ) else Real.sin (frequencies p.1 * θ)

theorem fourierCode_inner {m : ℕ} (weights frequencies : Fin m → ℝ) (θ φ : ℝ) :
    inner (𝕜 := ℝ) (fourierCode weights frequencies θ) (fourierCode weights frequencies φ) =
      ∑ k, weights k ^ 2 * Real.cos (frequencies k * (θ - φ)) := by
  rw [PiLp.inner_apply, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp [fourierCode, Fin.sum_univ_two, RCLike.inner_apply, Real.cos_sub, mul_sub]
  ring

theorem fourierCode_norm_sq {m : ℕ} (weights frequencies : Fin m → ℝ) (θ : ℝ) :
    ‖fourierCode weights frequencies θ‖ ^ 2 = ∑ k, weights k ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, fourierCode_inner]
  simp

theorem fourierCode_unit {m : ℕ} (weights frequencies : Fin m → ℝ)
    (hw : ∑ k, weights k ^ 2 = 1) (θ : ℝ) : ‖fourierCode weights frequencies θ‖ = 1 := by
  have h := fourierCode_norm_sq weights frequencies θ
  rw [hw] at h
  nlinarith [norm_nonneg (fourierCode weights frequencies θ)]

/-- Full real embeddings may form ellipses: equal orthogonal coefficient vectors
are exactly the useful sufficient conditions for a constant-norm frequency plane. -/
theorem harmonic_norm_sq {d : ℕ} (a b : EucSpace d) (θ : ℝ)
    (hab : inner (𝕜 := ℝ) a b = 0) (hequal : ‖a‖ = ‖b‖) :
    ‖Real.cos θ • a + Real.sin θ • b‖ ^ 2 = ‖a‖ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq]
  have hba : inner (𝕜 := ℝ) b a = 0 := by rw [real_inner_comm, hab]
  simp only [inner_add_left, inner_add_right, inner_smul_left, inner_smul_right,
    conj_trivial, hab, hba, real_inner_self_eq_norm_sq, ← hequal]
  linear_combination ‖a‖ ^ 2 * Real.sin_sq_add_cos_sq θ

def IsEquiangular {d n : ℕ} (ρ : ℝ) (x : Fin n → EucSpace d) : Prop :=
  (∀ i, ‖x i‖ = 1) ∧ ∀ i j, i ≠ j → inner (𝕜 := ℝ) (x i) (x j) = ρ

def equiGram (n : ℕ) (ρ : ℝ) (i j : Fin n) : ℝ := if i = j then 1 else ρ

theorem equiangular_gram {d n : ℕ} (ρ : ℝ) (x : Fin n → EucSpace d)
    (hx : IsEquiangular ρ x) (i j : Fin n) :
    inner (𝕜 := ℝ) (x i) (x j) = equiGram n ρ i j := by
  by_cases hij : i = j
  · subst j
    simp [equiGram, hx.1]
  · simp [equiGram, hij, hx.2 i j hij]

theorem equiangular_overlap_le_one {d n : ℕ} (hn : 2 ≤ n) (ρ : ℝ)
    (x : Fin n → EucSpace d) (hx : IsEquiangular ρ x) : ρ ≤ 1 := by
  let i : Fin n := ⟨0, by omega⟩
  let j : Fin n := ⟨1, by omega⟩
  have h := real_inner_le_norm (x i) (x j)
  rw [hx.2 i j (by simp [i, j]), hx.1 i, hx.1 j] at h
  simpa using h

/-- The lower overlap bound follows from the squared norm of the vector sum. -/
theorem equiangular_overlap_lower {d n : ℕ} (hn : 2 ≤ n) (ρ : ℝ)
    (x : Fin n → EucSpace d) (hx : IsEquiangular ρ x) : -1 / (n - 1 : ℝ) ≤ ρ := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hnR1 : (1 : ℝ) < n := by exact_mod_cast (by omega : 1 < n)
  have hn1 : (0 : ℝ) < n - 1 := by linarith
  have hrow (i : Fin n) : ∑ j, equiGram n ρ i j = 1 + ((n : ℝ) - 1) * ρ := by
    have he : ∀ j, equiGram n ρ i j = ρ + if j = i then 1 - ρ else 0 := by
      intro j; by_cases h : i = j <;> simp [equiGram, h, eq_comm]
    calc
      _ = ∑ j : Fin n, (ρ + if j = i then 1 - ρ else 0) := Finset.sum_congr rfl fun j _ => he j
      _ = (n : ℝ) * ρ + (1 - ρ) := by simp [Finset.sum_add_distrib]
      _ = _ := by ring
  have hs : ‖∑ i, x i‖ ^ 2 = (n : ℝ) * (1 + ((n : ℝ) - 1) * ρ) := by
    rw [← real_inner_self_eq_norm_sq, sum_inner]
    simp_rw [inner_sum, equiangular_gram ρ x hx]
    simp_rw [hrow]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hnonneg : 0 ≤ 1 + ((n : ℝ) - 1) * ρ := by nlinarith [sq_nonneg ‖∑ i, x i‖]
  apply (div_le_iff₀ hn1).2
  nlinarith

theorem antipodal_inner {d : ℕ} (u : EucSpace d) (hu : ‖u‖ = 1) :
    inner (𝕜 := ℝ) u (-u) = -1 := by
  rw [inner_neg_right, real_inner_self_eq_norm_sq, hu]
  norm_num

example : IsEquiangular 0 (fun i : Fin 2 => EuclideanSpace.single i (1 : ℝ)) := by
  constructor
  · intro i; simp
  · intro i j hij; simp [EuclideanSpace.inner_single_left, hij]

example : ∑ _ : Fin 1, (1 : ℝ) ^ 2 = 1 := by simp

end Grokking
