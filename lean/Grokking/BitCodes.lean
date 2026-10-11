import Grokking.Basic

/-!
The leading bit coordinates of wide parity codes and the hypercube option.
Extra parity coordinates cannot remove injectivity of this leading block.
Narrow radix-packed codes are covered by Rust tests, not this wide-code theorem.
-/

open scoped BigOperators

namespace Grokking

def bitSign (b : Bool) : ℝ := if b then 1 else -1

theorem bitSign_injective : Function.Injective bitSign := by
  intro a b hab
  cases a <;> cases b <;> norm_num [bitSign] at *

noncomputable def bitCode (d : ℕ) (i : Fin (2 ^ d)) : EucSpace d :=
  WithLp.toLp 2 fun k => bitSign (i.val.testBit k.val)

theorem bitCode_norm_sq (d : ℕ) (i : Fin (2 ^ d)) : ‖bitCode d i‖ ^ 2 = d := by
  rw [← real_inner_self_eq_norm_sq, PiLp.inner_apply]
  have he (k : Fin d) : inner (𝕜 := ℝ) (bitCode d i k) (bitCode d i k) = 1 := by
    simp only [bitCode, RCLike.inner_apply, conj_trivial]
    cases i.val.testBit k.val <;> simp [bitSign]
  simp_rw [he]
  simp

theorem bitCode_injective (d : ℕ) : Function.Injective (bitCode d) := by
  intro i j hij
  apply Fin.ext
  apply Nat.eq_of_testBit_eq
  intro k
  by_cases hk : k < d
  · exact bitSign_injective (congrArg (fun x : EucSpace d => x ⟨k, hk⟩) hij)
  · have hpow : 2 ^ d ≤ 2 ^ k := Nat.pow_le_pow_right (by decide) (by omega)
    rw [Nat.testBit_lt_two_pow (lt_of_lt_of_le i.isLt hpow),
      Nat.testBit_lt_two_pow (lt_of_lt_of_le j.isLt hpow)]

/-- Appending any features to the injective bit block preserves distinct tokens. -/
theorem extended_code_injective {d w : ℕ} (hd : d ≤ w) (code : Fin (2 ^ d) → EucSpace w)
    (hcode : ∀ i k, code i (Fin.castLE hd k) = bitCode d i k) : Function.Injective code := by
  intro i j hij
  apply bitCode_injective d
  ext k
  rw [← hcode i k, ← hcode j k, hij]

example : Function.Injective (bitCode 3) := bitCode_injective 3

end Grokking
