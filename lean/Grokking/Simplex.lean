import Grokking.Shapes

/-!
Canonical regular simplex geometry, in the zero-sum hyperplane of R^n.
The Rust initializer uses a Helmert orthonormal chart of this hyperplane in
R^(n-1), whose expected Gram is checked by the numerical construction tests.
-/

open scoped BigOperators

namespace Grokking

noncomputable def simplexCenter (n : ℕ) : EucSpace n :=
  WithLp.toLp 2 fun _ => 1 / n

noncomputable def simplexVertex (n : ℕ) (i : Fin n) : EucSpace n :=
  Real.sqrt ((n : ℝ) / (n - 1)) • (EuclideanSpace.single i 1 - simplexCenter n)

theorem simplexCenter_inner {n : ℕ} (hn : 0 < n) :
    inner (𝕜 := ℝ) (simplexCenter n) (simplexCenter n) = 1 / n := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_zero_of_lt hn)
  rw [PiLp.inner_apply]
  simp [simplexCenter]
  field_simp

theorem simplexVertex_inner {n : ℕ} (hn : 2 ≤ n) (i j : Fin n) :
    inner (𝕜 := ℝ) (simplexVertex n i) (simplexVertex n j) =
      if i = j then 1 else -1 / (n - 1 : ℝ) := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hn1 : (0 : ℝ) < (n : ℝ) - 1 := by
    have h : (1 : ℝ) < n := by exact_mod_cast (by omega : 1 < n)
    linarith
  have hc := simplexCenter_inner (by omega : 0 < n)
  rw [simplexVertex, simplexVertex, inner_smul_left, inner_smul_right]
  simp only [conj_trivial]
  rw [show Real.sqrt ((n : ℝ) / (n - 1)) *
      (Real.sqrt ((n : ℝ) / (n - 1)) * inner (𝕜 := ℝ)
        (EuclideanSpace.single i 1 - simplexCenter n) (EuclideanSpace.single j 1 - simplexCenter n)) =
      (Real.sqrt ((n : ℝ) / (n - 1))) ^ 2 * inner (𝕜 := ℝ)
        (EuclideanSpace.single i 1 - simplexCenter n) (EuclideanSpace.single j 1 - simplexCenter n) by ring]
  rw [Real.sq_sqrt (div_nonneg hnR.le hn1.le), inner_sub_left, inner_sub_right, inner_sub_right]
  rw [hc]
  simp only [EuclideanSpace.inner_single_left, EuclideanSpace.inner_single_right,
    map_one, one_mul, simplexCenter]
  by_cases hij : i = j
  · subst j
    simp
    field_simp
  · simp [hij]
    field_simp

theorem simplexVertex_equiangular {n : ℕ} (hn : 2 ≤ n) :
    IsEquiangular (-1 / (n - 1 : ℝ)) (simplexVertex n) := by
  constructor
  · intro i
    have h := simplexVertex_inner hn i i
    rw [real_inner_self_eq_norm_sq, ite_eq_left rfl] at h
    nlinarith [norm_nonneg (simplexVertex n i)]
  · intro i j hij
    simpa [hij] using simplexVertex_inner hn i j

example : IsEquiangular (-1 : ℝ) (simplexVertex 2) := by
  have h := simplexVertex_equiangular (by norm_num : 2 ≤ 2)
  norm_num at h
  exact h

end Grokking
