/-
# Sphere packing — the cardinality bounds

The two volume ledgers are inequalities between `(1 ± δ/2)^d` and
`card S · (δ/2)^d`.  Reading a bound on `card S` off them means bounding the
*shell* `x^d - y^d` from both sides, which the factorization

  `x^d - y^d = (∑_{i<d} x^i y^{d-1-i}) (x - y)`

does with no analysis: each of the `d` terms of the sum lies between `y^{d-1}`
and `x^{d-1}`.  The outcome is the packing number of `𝕊^{d-1}` at scale `δ`,

  `d (1/(4δ))^{d-1} ≤ card S ≤ 2 d (4/δ)^{d-1}`,

the first for a maximal separated set, the second for any separated set.
-/

import Grokking.Packing.Lower

open Metric

namespace Grokking
namespace Packing

/-- Auxiliary (not from the paper): `x^d - y^d ≤ d x^{d-1} (x - y)` for
`0 ≤ y ≤ x` — the mean value inequality, in the algebraic form the
factorization of `x^d - y^d` gives for free. -/
theorem pow_sub_pow_le (x y : ℝ) (hy : 0 ≤ y) (hxy : y ≤ x) (d : ℕ) :
    x ^ d - y ^ d ≤ (d : ℝ) * x ^ (d - 1) * (x - y) := by
  have hx : 0 ≤ x := le_trans hy hxy
  rw [← geom_sum₂_mul x y d]
  refine mul_le_mul_of_nonneg_right ?_ (by linarith)
  have hterm : ∀ i ∈ Finset.range d, x ^ i * y ^ (d - 1 - i) ≤ x ^ (d - 1) := by
    intro i hi
    have hi' : i ≤ d - 1 := by simp only [Finset.mem_range] at hi; omega
    calc x ^ i * y ^ (d - 1 - i) ≤ x ^ i * x ^ (d - 1 - i) :=
          mul_le_mul_of_nonneg_left (pow_le_pow_left₀ hy hxy _) (pow_nonneg hx i)
      _ = x ^ (d - 1) := by rw [← pow_add]; congr 1; omega
  calc (∑ i ∈ Finset.range d, x ^ i * y ^ (d - 1 - i))
      ≤ (Finset.range d).card • x ^ (d - 1) := Finset.sum_le_card_nsmul _ _ _ hterm
    _ = (d : ℝ) * x ^ (d - 1) := by rw [Finset.card_range, nsmul_eq_mul]

/-- The hypotheses of `pow_sub_pow_le` are satisfiable: `x = 1`, `y = 0`. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := ⟨le_rfl, zero_le_one⟩

/-- Auxiliary (not from the paper): the matching lower bound
`d y^{d-1} (x - y) ≤ x^d - y^d` for `0 ≤ y ≤ x`. -/
theorem le_pow_sub_pow (x y : ℝ) (hy : 0 ≤ y) (hxy : y ≤ x) (d : ℕ) :
    (d : ℝ) * y ^ (d - 1) * (x - y) ≤ x ^ d - y ^ d := by
  rw [← geom_sum₂_mul x y d]
  refine mul_le_mul_of_nonneg_right ?_ (by linarith)
  have hterm : ∀ i ∈ Finset.range d, y ^ (d - 1) ≤ x ^ i * y ^ (d - 1 - i) := by
    intro i hi
    have hi' : i ≤ d - 1 := by simp only [Finset.mem_range] at hi; omega
    calc y ^ (d - 1) = y ^ i * y ^ (d - 1 - i) := by rw [← pow_add]; congr 1; omega
      _ ≤ x ^ i * y ^ (d - 1 - i) :=
          mul_le_mul_of_nonneg_right (pow_le_pow_left₀ hy hxy _) (pow_nonneg hy _)
  calc (d : ℝ) * y ^ (d - 1) = (Finset.range d).card • y ^ (d - 1) := by
        rw [Finset.card_range, nsmul_eq_mul]
    _ ≤ ∑ i ∈ Finset.range d, x ^ i * y ^ (d - 1 - i) := Finset.card_nsmul_le_sum _ _ _ hterm

/-- The hypotheses of `le_pow_sub_pow` are satisfiable: `x = 1`, `y = 0`. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := ⟨le_rfl, zero_le_one⟩

/-- **Packing number of the sphere, upper bound.**  A `δ`-separated set of unit
vectors of `EucSpace d` has at most `2 d (4/δ)^{d-1}` elements: the shell
`1 - δ/2 < ‖y‖ < 1 + δ/2` has volume `≍ d δ`, and it holds `card S` disjoint
balls of radius `δ/2`.

Auxiliary (not from the paper): the sharp form of the packing bound behind the
cardinality claim of §5 of arXiv:2411.04990v2. -/
theorem card_le_of_separatedOnSphere (d : ℕ) (hd : 1 ≤ d) (δ : ℝ) (hδ : 0 < δ) (hδ2 : δ ≤ 2)
    (S : Finset (EucSpace d)) (hS : SeparatedOnSphere d S δ) :
    (S.card : ℝ) ≤ 2 * d * (4 / δ) ^ (d - 1) := by
  obtain ⟨m, rfl⟩ : ∃ m : ℕ, d = m + 1 := ⟨d - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  have h1 := separated_volume_ledger (m + 1) (by omega) δ hδ hδ2 S hS
  have h2 := pow_sub_pow_le (1 + δ / 2) (1 - δ / 2) (by linarith) (by linarith) (m + 1)
  simp only [Nat.add_sub_cancel] at h2
  push_cast at h2
  have h3 : ((m : ℝ) + 1) * (1 + δ / 2) ^ m * ((1 + δ / 2) - (1 - δ / 2))
      ≤ ((m : ℝ) + 1) * 2 ^ m * δ := by
    have hpow : (1 + δ / 2) ^ m ≤ 2 ^ m := pow_le_pow_left₀ (by linarith) (by linarith) m
    have hstep : ((m : ℝ) + 1) * (1 + δ / 2) ^ m ≤ ((m : ℝ) + 1) * 2 ^ m :=
      mul_le_mul_of_nonneg_left hpow (by positivity)
    calc ((m : ℝ) + 1) * (1 + δ / 2) ^ m * ((1 + δ / 2) - (1 - δ / 2))
        = ((m : ℝ) + 1) * (1 + δ / 2) ^ m * δ := by ring
      _ ≤ ((m : ℝ) + 1) * 2 ^ m * δ := mul_le_mul_of_nonneg_right hstep hδ.le
  have hlow : (0 : ℝ) ≤ (1 - δ / 2) ^ (m + 1) := pow_nonneg (by linarith) _
  have key : (S.card : ℝ) * (δ / 2) ^ (m + 1) ≤ ((m : ℝ) + 1) * 2 ^ m * δ := by linarith
  have hpos : (0 : ℝ) < (δ / 2) ^ (m + 1) := by positivity
  push_cast
  rw [show 2 * ((m : ℝ) + 1) * (4 / δ) ^ m = (((m : ℝ) + 1) * 2 ^ m * δ) / (δ / 2) ^ (m + 1) by
    rw [div_pow, div_pow, show (4 : ℝ) ^ m = 2 ^ m * 2 ^ m by rw [← mul_pow]; norm_num]
    field_simp
    ring, le_div_iff₀ hpos]
  exact key

/-- The hypotheses of `card_le_of_separatedOnSphere` are satisfiable: `d = 1`,
`δ = 1`, `S = ∅`. -/
example : SeparatedOnSphere 1 (∅ : Finset (EucSpace 1)) 1 :=
  ⟨fun _ hx => absurd hx (Finset.notMem_empty _), fun _ hx => absurd hx (Finset.notMem_empty _)⟩

/-- **Packing number of the sphere, lower bound.**  A *maximal* `δ`-separated
set of unit vectors of `EucSpace d` has at least `d (1/(4δ))^{d-1}` elements:
its `2δ`-balls cover the shell `1 - δ < ‖y‖ < 1 + δ`, of volume `≍ d δ`.

Auxiliary (not from the paper): the sharp form of the covering bound behind the
cardinality claim of §5 of arXiv:2411.04990v2. -/
theorem card_ge_of_maximalSeparated (d : ℕ) (hd : 1 ≤ d) (δ : ℝ) (hδ : 0 < δ) (hδ2 : δ ≤ 1 / 2)
    (S : Finset (EucSpace d)) (hS : IsMaximalSeparated d S δ) :
    (d : ℝ) * (1 / (4 * δ)) ^ (d - 1) ≤ (S.card : ℝ) := by
  obtain ⟨m, rfl⟩ : ∃ m : ℕ, d = m + 1 := ⟨d - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  have h1 := maximal_volume_ledger (m + 1) (by omega) δ hδ (by linarith) S hS
  have h2 := le_pow_sub_pow (1 + δ) (1 - δ) (by linarith) (by linarith) (m + 1)
  simp only [Nat.add_sub_cancel] at h2
  push_cast at h2
  have h3 : ((m : ℝ) + 1) * (1 / 2) ^ m * (2 * δ)
      ≤ ((m : ℝ) + 1) * (1 - δ) ^ m * ((1 + δ) - (1 - δ)) := by
    have hpow : ((1 : ℝ) / 2) ^ m ≤ (1 - δ) ^ m := pow_le_pow_left₀ (by norm_num) (by linarith) m
    have hstep : ((m : ℝ) + 1) * (1 / 2) ^ m ≤ ((m : ℝ) + 1) * (1 - δ) ^ m :=
      mul_le_mul_of_nonneg_left hpow (by positivity)
    calc ((m : ℝ) + 1) * (1 / 2) ^ m * (2 * δ)
        ≤ ((m : ℝ) + 1) * (1 - δ) ^ m * (2 * δ) :=
          mul_le_mul_of_nonneg_right hstep (by linarith)
      _ = ((m : ℝ) + 1) * (1 - δ) ^ m * ((1 + δ) - (1 - δ)) := by ring
  have key : ((m : ℝ) + 1) * (1 / 2) ^ m * (2 * δ) ≤ (S.card : ℝ) * (2 * δ) ^ (m + 1) := by
    linarith
  have hpos : (0 : ℝ) < (2 * δ) ^ (m + 1) := by positivity
  push_cast
  rw [show ((m : ℝ) + 1) * (1 / (4 * δ)) ^ m
      = (((m : ℝ) + 1) * (1 / 2) ^ m * (2 * δ)) / (2 * δ) ^ (m + 1) by
    rw [div_pow, div_pow, mul_pow, show (4 : ℝ) ^ m = 2 ^ m * 2 ^ m by rw [← mul_pow]; norm_num]
    field_simp
    ring, div_le_iff₀ hpos]
  exact key

/-- The hypotheses of `card_ge_of_maximalSeparated` are satisfiable: `d = 1`,
`δ = 1/2`, and a maximal set at that scale exists by
`exists_maximalSeparated`. -/
example : ∃ S : Finset (EucSpace 1), IsMaximalSeparated 1 S (1 / 2) :=
  exists_maximalSeparated 1 le_rfl (1 / 2) (by norm_num) (by norm_num)

end Packing
end Grokking
