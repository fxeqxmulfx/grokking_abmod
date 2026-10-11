/-
# Kinetic theory for Transformers — the vocabulary's codewords

The geometry the retrieval task of `eq:Acc-def` of arXiv:2605.09213v1,
*Kinetic theory for Transformers and the lost-in-the-middle phenomenon*, §1.1
rests on: the codewords `ϑ_m = 2πm/M` of a vocabulary of size `M`, the torus
distance the decoder minimizes, and the two facts that make the decoder a
radius test — the codewords are `2π/M` apart, and every angle of `𝕋` is within
`π/M` of one of them.

Everything here is proved.  `Accuracy.lean` turns it into the identity of
`eq:Acc-def`.

Source: arXiv:2605.09213v1, §1.1.
-/

import Grokking.Basic

open scoped BigOperators
open Real

namespace Grokking
namespace Codewords

/-- The torus distance `|a - b|_𝕋` between two angles, as the distance of their
classes in `𝕋 = ℝ/2πℤ`.

Source: arXiv:2605.09213v1, §1.1. -/
noncomputable def torusDist (a b : ℝ) : ℝ := dist ((a : Torus)) ((b : Torus))

/-- The codeword `ϑ_m = 2πm/M ∈ 𝕋` of the `m`-th word of a vocabulary of size
`M`.

Source: arXiv:2605.09213v1, §1.1. -/
noncomputable def codewordAngle (M m : ℕ) : ℝ := 2 * π * m / M

theorem torusDist_comm (a b : ℝ) : torusDist a b = torusDist b a := dist_comm _ _

theorem torusDist_nonneg (a b : ℝ) : 0 ≤ torusDist a b := dist_nonneg

theorem torusDist_triangle (a b c : ℝ) : torusDist a c ≤ torusDist a b + torusDist b c :=
  dist_triangle _ _ _

/-- The torus distance is at most the distance on `ℝ`: passing to the quotient
can only shorten. -/
theorem torusDist_le_abs (a b : ℝ) : torusDist a b ≤ |a - b| := by
  rw [torusDist, dist_eq_norm, ← QuotientAddGroup.mk_sub]
  exact QuotientAddGroup.norm_mk_le_norm.trans_eq (Real.norm_eq_abs _)

/-- Shifting by a whole number of periods does not move a point of `𝕋`. -/
theorem coe_add_int_mul_period (x : ℝ) (n : ℤ) :
    ((x + n * (2 * π) : ℝ) : Torus) = (x : Torus) := by
  rw [QuotientAddGroup.mk_add]
  have hz : ((n * (2 * π) : ℝ) : Torus) = 0 := by
    rw [AddCircle.coe_eq_zero_iff]
    exact ⟨n, by simp [zsmul_eq_mul]⟩
  rw [hz, add_zero]

/-- **The codewords are `2π/M` apart.**  Distinct words of a vocabulary of size
`M` are encoded at torus distance at least `2π/M`; this is what makes the
nearest-codeword decoder of `eq:Acc-def` correct below half that radius.

Source: arXiv:2605.09213v1, §1.1. -/
theorem two_pi_div_le_torusDist_codeword {M j k : ℕ} (hj : j < M) (hk : k < M) (hjk : j ≠ k) :
    2 * π / M ≤ torusDist (codewordAngle M j) (codewordAngle M k) := by
  wlog h : k < j generalizing j k
  · rw [torusDist_comm]
    exact this hk hj (Ne.symm hjk) (by omega)
  have hM : 0 < M := lt_of_le_of_lt (Nat.zero_le _) hj
  have hMR : (0 : ℝ) < M := by exact_mod_cast hM
  have hdiff : codewordAngle M j - codewordAngle M k = 2 * π * ((j - k : ℕ) : ℝ) / M := by
    rw [codewordAngle, codewordAngle, Nat.cast_sub h.le]
    field_simp
  rw [torusDist, dist_eq_norm, ← QuotientAddGroup.mk_sub, hdiff]
  have hnorm : ‖((2 * π * ((j - k : ℕ) : ℝ) / M : ℝ) : Torus)‖
      = 2 * π * (((min ((j - k) % M) (M - (j - k) % M) : ℕ) : ℝ) / M) := by
    have hcast : (2 * π * ((j - k : ℕ) : ℝ) / M : ℝ) = ((j - k : ℕ) : ℝ) / (M : ℝ) * (2 * π) := by
      ring
    rw [hcast]
    exact AddCircle.norm_div_natCast
  rw [hnorm, Nat.mod_eq_of_lt (by omega : j - k < M)]
  have hone : (1 : ℝ) ≤ ((min (j - k) (M - (j - k)) : ℕ) : ℝ) := by
    have : 1 ≤ min (j - k) (M - (j - k)) := by omega
    exact_mod_cast this
  calc 2 * π / M = 2 * π * ((1 : ℝ) / M) := by ring
    _ ≤ 2 * π * (((min (j - k) (M - (j - k)) : ℕ) : ℝ) / M) := by gcongr

/-- The hypotheses of `two_pi_div_le_torusDist_codeword` are satisfiable. -/
example : 0 < 2 ∧ 1 < 2 ∧ (0 : ℕ) ≠ 1 := ⟨by norm_num, by norm_num, by norm_num⟩

/-- **Every angle has a codeword within `π/M`.**  The `M` codewords cover `𝕋`
at radius `π/M`, so the decoder of `eq:Acc-def` never has to reach further.

Source: arXiv:2605.09213v1, §1.1. -/
theorem exists_codeword_close {M : ℕ} (hM : 0 < M) (θ : ℝ) :
    ∃ j < M, torusDist θ (codewordAngle M j) ≤ π / M := by
  have hMZ : (0 : ℤ) < M := by exact_mod_cast hM
  have hMR : (0 : ℝ) < M := by exact_mod_cast hM
  set r : ℤ := round ((M : ℝ) * θ / (2 * π)) with hr
  have hnn : 0 ≤ r % M := Int.emod_nonneg r (ne_of_gt hMZ)
  have hlt : r % M < M := Int.emod_lt_of_pos r hMZ
  refine ⟨(r % M).toNat, by omega, ?_⟩
  have hjr : (((r % M).toNat : ℕ) : ℤ) = r % M := Int.toNat_of_nonneg hnn
  have hshift : codewordAngle M ((r % M).toNat)
      = 2 * π * (r : ℝ) / M + (-(r / M) : ℤ) * (2 * π) := by
    have hq : (r : ℝ) = (M : ℝ) * ((r / M : ℤ) : ℝ) + (((r % M).toNat : ℕ) : ℝ) := by
      have hz : (M : ℤ) * (r / (M : ℤ)) + (((r % (M : ℤ)).toNat : ℕ) : ℤ) = r := by
        rw [hjr]
        exact Int.mul_ediv_add_emod r (M : ℤ)
      exact_mod_cast hz.symm
    rw [codewordAngle, hq]
    push_cast
    field_simp
    ring
  have hcong : ((codewordAngle M ((r % M).toNat) : ℝ) : Torus)
      = ((2 * π * (r : ℝ) / M : ℝ) : Torus) := by
    rw [hshift, coe_add_int_mul_period]
  have hbound : |θ - 2 * π * (r : ℝ) / M| ≤ π / M := by
    have hfac : θ - 2 * π * (r : ℝ) / M
        = (2 * π / M) * ((M : ℝ) * θ / (2 * π) - (r : ℝ)) := by
      field_simp
    rw [hfac, abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 2 * π / M)]
    calc 2 * π / M * |(M : ℝ) * θ / (2 * π) - (r : ℝ)|
        ≤ 2 * π / M * (1 / 2) := by
          gcongr
          exact abs_sub_round _
      _ = π / M := by ring
  calc torusDist θ (codewordAngle M ((r % M).toNat))
      = dist ((θ : ℝ) : Torus) ((2 * π * (r : ℝ) / M : ℝ) : Torus) := by rw [torusDist, hcong]
    _ ≤ |θ - 2 * π * (r : ℝ) / M| := torusDist_le_abs _ _
    _ ≤ π / M := hbound

/-- The hypothesis of `exists_codeword_close` is satisfiable. -/
example : 0 < 1 := Nat.one_pos

/-- **The decoder of `eq:Acc-def`:** `m̂ = argmin_{m ∈ 𝒱} |θ - ϑ_m|_𝕋`, written
as "`m` is a word of the vocabulary and no word is closer".  `argmin` over a
finite non-empty set returns a minimizer; which one it returns on a tie is not
determined, and this predicate says exactly that much.

Source: arXiv:2605.09213v1, §1.1. -/
def IsNearestCodeword (M : ℕ) (θ : ℝ) (m : ℕ) : Prop :=
  m < M ∧ ∀ k < M, torusDist θ (codewordAngle M m) ≤ torusDist θ (codewordAngle M k)

/-- **The content of `eq:Acc-def`.**  Away from a tie, the decoder returns `m`
exactly when the hidden state is within `π/M` of the codeword `ϑ_m`.

Source: arXiv:2605.09213v1, `eq:Acc-def`. -/
theorem isNearestCodeword_iff {M : ℕ} (hM : 0 < M) {θ : ℝ} {m mhat : ℕ} (hm : m < M)
    (hmhat : IsNearestCodeword M θ mhat)
    (hne : torusDist θ (codewordAngle M m) ≠ π / M) :
    mhat = m ↔ torusDist θ (codewordAngle M m) ≤ π / M := by
  constructor
  · rintro rfl
    obtain ⟨j, hj, hjle⟩ := exists_codeword_close hM θ
    exact (hmhat.2 j hj).trans hjle
  · intro hle
    have hlt : torusDist θ (codewordAngle M m) < π / M := lt_of_le_of_ne hle hne
    by_contra hne'
    have h1 : 2 * π / M ≤ torusDist (codewordAngle M m) (codewordAngle M mhat) :=
      two_pi_div_le_torusDist_codeword hm hmhat.1 (Ne.symm hne')
    have h2 : torusDist (codewordAngle M m) (codewordAngle M mhat)
        ≤ torusDist (codewordAngle M m) θ + torusDist θ (codewordAngle M mhat) :=
      torusDist_triangle _ _ _
    have h3 : torusDist θ (codewordAngle M mhat) ≤ torusDist θ (codewordAngle M m) :=
      hmhat.2 m hm
    rw [torusDist_comm (codewordAngle M m) θ] at h2
    have hsplit : 2 * π / (M : ℝ) = 2 * (π / (M : ℝ)) := by ring
    linarith

/-- The hypotheses of `isNearestCodeword_iff` are satisfiable: a vocabulary of
one word, whose single codeword is `0`, and a hidden state sitting on it. -/
example : IsNearestCodeword 1 0 0 ∧ torusDist 0 (codewordAngle 1 0) ≠ π / 1 := by
  refine ⟨⟨Nat.one_pos, fun k hk => by interval_cases k; exact le_refl _⟩, ?_⟩
  simp only [torusDist, codewordAngle, Nat.cast_zero, Nat.cast_one, div_one, mul_zero,
    dist_self]
  exact fun h => absurd h.symm (ne_of_gt Real.pi_pos)

end Codewords
end Grokking
