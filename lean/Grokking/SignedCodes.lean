import Grokking.Basic

/-!
# Compact integer codes for all 256 Basis recall keys

New capacity construction for the repaired genuine matching architecture
after arXiv:2211.11052v1, §3.1. Signed permutations of (1,2,3,4) have
coordinate cap four and squared norm thirty. Sixteen distinct permutations
and sixteen sign patterns give 256 different four-dimensional key codes.
Later equal-norm integer geometry supplies the unit sparsemax score gap
of arXiv:1602.02068v2, §2.2, without increasing physical matching width.

The small sixteen-element magnitude and sign checks below are reduced by
Lean's kernel (`decide`), not native evaluation. The full 256-key properties
follow algebraically from those factors and quotient/remainder decoding.
This constructs a feasible parameter table, not an optimizer convergence
guarantee. The numerical architecture leaves Q and K freely trainable.
-/

namespace Grokking.SignedCodes

open scoped BigOperators

/-- Rotate two independent adjacent-pair swaps, giving sixteen different permutations.
Source: the new signed-permutation capacity construction after §3.1. -/
def pairedRecallMagnitude (p : Fin 16) (d : Fin 4) : ℤ :=
  let slot := if d.val < 2 then (d.val + p.val / 2 % 2) % 2
    else 2 + (d.val - 2 + p.val % 2) % 2
  (1 + (slot + p.val / 4) % 4 : ℕ)

/-- Select the permutation factor of the compact 256-key code.
Source: the sixteen-by-sixteen indexing of the new §3.1 construction. -/
def pairedRecallPermutation (i : Fin 256) : Fin 16 := ⟨i.val / 16, by omega⟩

/-- Select the sign-pattern factor without adding physical embedding dimensions.
Source: the sixteen-by-sixteen indexing of the new §3.1 construction. -/
def pairedRecallSigns (i : Fin 256) : Fin 16 := ⟨i.val % 16, Nat.mod_lt _ (by decide)⟩

/-- A representable integer Q/K witness with no learned token-pair parameter table.
Source: the new compact signed-permutation binding construction after §3.1. -/
def pairedRecallCode (i : Fin 256) (d : Fin 4) : ℤ :=
  let magnitude := pairedRecallMagnitude (pairedRecallPermutation i) d
  if (pairedRecallSigns i).val.testBit d.val then -magnitude else magnitude

/-- Every unsigned coordinate is between one and four, independently of its permutation.
Source: the kernel-checked finite magnitude factor of the new compact key construction. -/
theorem pairedRecallMagnitude_bounds (p : Fin 16) (d : Fin 4) :
    1 ≤ pairedRecallMagnitude p d ∧ pairedRecallMagnitude p d ≤ 4 := by
  have hc : ∀ p : Fin 16, ∀ d : Fin 4,
      1 ≤ pairedRecallMagnitude p d ∧ pairedRecallMagnitude p d ≤ 4 := by decide
  exact hc p d

/-- All sixteen permutation factors have the same squared norm thirty.
Source: the kernel-checked finite magnitude factor of the new §3.1 construction. -/
theorem pairedRecallMagnitude_norm (p : Fin 16) : ∑ d, pairedRecallMagnitude p d ^ 2 = 30 := by
  have hn : ∀ p : Fin 16, ∑ d, pairedRecallMagnitude p d ^ 2 = 30 := by decide
  exact hn p

/-- The sixteen magnitude permutations are distinct before applying signs.
Source: the kernel-checked finite magnitude factor of the new compact key construction. -/
theorem pairedRecallMagnitude_injective : Function.Injective pairedRecallMagnitude := by
  have hi : ∀ p q : Fin 16, (∀ d : Fin 4, pairedRecallMagnitude p d = pairedRecallMagnitude q d) →
      p = q := by decide
  intro p q hpq
  exact hi p q (congrFun hpq)

/-- Every signed key coordinate fits exactly the numerical cap four used in the experiment.
Source: the magnitude bounds of the new compact §3.1 key construction. -/
theorem pairedRecallCode_bounds (i : Fin 256) (d : Fin 4) :
    -4 ≤ pairedRecallCode i d ∧ pairedRecallCode i d ≤ 4 := by
  have hm := pairedRecallMagnitude_bounds (pairedRecallPermutation i) d
  unfold pairedRecallCode
  split_ifs <;> dsimp only <;> omega

/-- Absolute coordinates recover the magnitude permutation from the actual signed table.
Source: the factorized decoding of the new compact §3.1 key construction. -/
theorem pairedRecallCode_abs (i : Fin 256) (d : Fin 4) :
    |pairedRecallCode i d| = pairedRecallMagnitude (pairedRecallPermutation i) d := by
  have hm : 0 ≤ pairedRecallMagnitude (pairedRecallPermutation i) d := by
    have hb := pairedRecallMagnitude_bounds (pairedRecallPermutation i) d
    omega
  unfold pairedRecallCode
  split_ifs
  · rw [abs_neg, abs_of_nonneg hm]
  · exact abs_of_nonneg hm

/-- All 256 actual signed key codes have squared norm thirty.
Source: the sixteen magnitude norms and unchanged squares under sign flips after §3.1. -/
theorem pairedRecallCode_norm (i : Fin 256) : ∑ d, pairedRecallCode i d ^ 2 = 30 := by
  have he (d : Fin 4) : pairedRecallCode i d ^ 2 = pairedRecallMagnitude (pairedRecallPermutation i) d ^ 2 := by
    unfold pairedRecallCode
    split_ifs <;> ring
  calc
    _ = ∑ d, pairedRecallMagnitude (pairedRecallPermutation i) d ^ 2 :=
      Finset.sum_congr rfl (fun d hd => he d)
    _ = 30 := pairedRecallMagnitude_norm _

/-- Magnitudes and four sign bits jointly distinguish all 256 original key identities.
Source: the small finite factor checks plus arithmetic decoding of the new compact code. -/
theorem pairedRecallCode_injective : Function.Injective pairedRecallCode := by
  have hsign : ∀ a b : Fin 16, (∀ d : Fin 4, a.val.testBit d.val = b.val.testBit d.val) → a = b := by
    decide
  intro i j hij
  have hp : pairedRecallPermutation i = pairedRecallPermutation j := by
    apply pairedRecallMagnitude_injective
    funext d
    have ha := congrArg abs (congrFun hij d)
    simpa only [pairedRecallCode_abs] using ha
  have hs : pairedRecallSigns i = pairedRecallSigns j := by
    apply hsign
    intro d
    have hc := congrFun hij d
    simp only [pairedRecallCode, hp] at hc
    have hm := (pairedRecallMagnitude_bounds (pairedRecallPermutation j) d).1
    cases hi : (pairedRecallSigns i).val.testBit d.val <;>
      cases hj : (pairedRecallSigns j).val.testBit d.val
    · rfl
    · simp [hi, hj] at hc
      omega
    · simp [hi, hj] at hc
      omega
    · rfl
  apply Fin.ext
  have hquot := congrArg Fin.val hp
  have hrem := congrArg Fin.val hs
  simp only [pairedRecallPermutation] at hquot
  simp only [pairedRecallSigns] at hrem
  omega

/-- The first code is the original positive vector of squared norm thirty.
Source: the explicit decoder of the new compact §3.1 construction. -/
theorem pairedRecallCode_zero : pairedRecallCode 0 = ![(1 : ℤ), 2, 3, 4] := by
  ext d
  fin_cases d <;> norm_num [pairedRecallCode, pairedRecallMagnitude, pairedRecallPermutation, pairedRecallSigns,
    Nat.testBit]

/-- The next identity changes one sign and remains a different eligible key code.
Source: the explicit decoder of the new compact §3.1 construction. -/
theorem pairedRecallCode_one : pairedRecallCode 1 = ![(-1 : ℤ), 2, 3, 4] := by
  ext d
  fin_cases d <;> norm_num [pairedRecallCode, pairedRecallMagnitude, pairedRecallPermutation, pairedRecallSigns,
    Nat.testBit]

/-- A second sign bit independently changes the second physical key coordinate.
Source: the explicit decoder of the new compact §3.1 construction. -/
theorem pairedRecallCode_two : pairedRecallCode 2 = ![(1 : ℤ), -2, 3, 4] := by
  ext d
  fin_cases d <;> norm_num [pairedRecallCode, pairedRecallMagnitude, pairedRecallPermutation, pairedRecallSigns,
    Nat.testBit]

end Grokking.SignedCodes
