/-
# Sphere packing — the lower ledger

A maximal `δ`-separated set `S` of unit vectors covers the sphere at scale `δ`,
hence covers the shell `1 - δ < ‖y‖ < 1 + δ` at scale `2 δ`: a point `z` of the
shell is within `δ` of `z / ‖z‖`, which is within `δ` of some `x ∈ S`.
Comparing volumes bounds `card S` from below.

As in `Packing.Upper`, the shell is handled without a set difference:
the inner ball stays in the ledger as one more piece.
-/

import Grokking.Packing.Upper

open MeasureTheory Metric
open scoped ENNReal

namespace Grokking
namespace Packing

/-- **Packing ledger (lower).**  A maximal `δ`-separated set `S` of unit
vectors covers the sphere at scale `δ`, hence the shell
`1 - δ < ‖y‖ < 1 + δ` at scale `2 δ`: every `z` of the shell is within `2 δ`
of the point of `S` nearest to `z / ‖z‖`.  Adding the inner ball back,

  `(1 + δ)^d ≤ (1 - δ)^d + card S · (2 δ)^d`.

Auxiliary (not from the paper): the volume form of the lower packing bound
behind the cardinality claim of §5. -/
theorem maximal_volume_ledger (d : ℕ) (hd : 1 ≤ d) (δ : ℝ) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (S : Finset (EucSpace d)) (hS : IsMaximalSeparated d S δ) :
    (1 + δ) ^ d ≤ (1 - δ) ^ d + (S.card : ℝ) * (2 * δ) ^ d := by
  have hsub : ball (0 : EucSpace d) (1 + δ)
      ⊆ closedBall (0 : EucSpace d) (1 - δ) ∪ ⋃ x ∈ S, closedBall x (2 * δ) := by
    intro z hz
    have hz1 : ‖z‖ < 1 + δ := mem_ball_zero_iff.mp hz
    by_cases hin : ‖z‖ ≤ 1 - δ
    · exact Set.mem_union_left _ (mem_closedBall_zero_iff.mpr hin)
    push Not at hin
    have hzn : 0 < ‖z‖ := by linarith
    have hne : ‖z‖ ≠ 0 := hzn.ne'
    set y := (‖z‖)⁻¹ • z with hy
    have hy1 : ‖y‖ = 1 := by
      rw [hy, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hne]
    obtain ⟨x, hxS, hyx⟩ := hS.2 y hy1
    have hzy : ‖z - y‖ = |‖z‖ - 1| := by
      have hzz : z - y = (1 - (‖z‖)⁻¹) • z := by rw [hy, sub_smul, one_smul]
      rw [hzz, norm_smul, Real.norm_eq_abs,
        show |1 - (‖z‖)⁻¹| * ‖z‖ = |(1 - (‖z‖)⁻¹) * ‖z‖| by
          rw [abs_mul, abs_of_nonneg (norm_nonneg z)]]
      congr 1
      field_simp
    have hzy' : ‖z - y‖ < δ := by
      rw [hzy, abs_lt]
      constructor <;> linarith
    refine Set.mem_union_right _ ?_
    simp only [Set.mem_iUnion, exists_prop]
    refine ⟨x, hxS, ?_⟩
    rw [mem_closedBall, dist_eq_norm]
    calc ‖z - x‖ ≤ ‖z - y‖ + ‖y - x‖ := by
          simpa using norm_add_le (z - y) (y - x)
      _ ≤ 2 * δ := by linarith
  have hle : volume (ball (0 : EucSpace d) (1 + δ))
      ≤ volume (closedBall (0 : EucSpace d) (1 - δ)) + ∑ x ∈ S, volume (closedBall x (2 * δ)) :=
    calc volume (ball (0 : EucSpace d) (1 + δ))
        ≤ volume (closedBall (0 : EucSpace d) (1 - δ) ∪ ⋃ x ∈ S, closedBall x (2 * δ)) :=
          measure_mono hsub
      _ ≤ volume (closedBall (0 : EucSpace d) (1 - δ)) + volume (⋃ x ∈ S, closedBall x (2 * δ)) :=
          measure_union_le _ _
      _ ≤ volume (closedBall (0 : EucSpace d) (1 - δ)) + ∑ x ∈ S, volume (closedBall x (2 * δ)) :=
          add_le_add le_rfl (measure_biUnion_finset_le _ _)
  rw [volume_ball_eucSpace d hd _ (by linarith : (0 : ℝ) ≤ 1 + δ),
    volume_closedBall_eucSpace d hd _ (by linarith : (0 : ℝ) ≤ 1 - δ),
    Finset.sum_congr rfl
      (fun x _ => volume_closedBall_eucSpace d hd x (by linarith : (0 : ℝ) ≤ 2 * δ)),
    Finset.sum_const, nsmul_eq_mul] at hle
  have hle2 : ENNReal.ofReal ((1 + δ) ^ d) * volume (ball (0 : EucSpace d) 1)
      ≤ (ENNReal.ofReal ((1 - δ) ^ d) + (S.card : ℝ≥0∞) * ENNReal.ofReal ((2 * δ) ^ d))
        * volume (ball (0 : EucSpace d) 1) := by
    rw [add_mul, mul_assoc]
    exact hle
  have hcancel := (ENNReal.mul_le_mul_iff_left
    (measure_ball_pos volume (0 : EucSpace d) one_pos).ne' measure_ball_lt_top.ne).mp hle2
  rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (Nat.cast_nonneg _),
    ← ENNReal.ofReal_add (pow_nonneg (by linarith) d) (by positivity),
    ENNReal.ofReal_le_ofReal_iff (by positivity)] at hcancel
  exact hcancel

/-- The hypotheses of `maximal_volume_ledger` are satisfiable: `d = 1`,
`δ = 1`, and a maximal set at that scale exists by `exists_maximalSeparated`. -/
example : ∃ S : Finset (EucSpace 1), IsMaximalSeparated 1 S 1 :=
  exists_maximalSeparated 1 le_rfl 1 one_pos (by norm_num)

end Packing
end Grokking
