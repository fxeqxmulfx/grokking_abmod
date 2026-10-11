/-
# Sphere packing — the upper ledger

A `δ`-separated set `S` of unit vectors carries disjoint balls `B(x, δ/2)`,
`x ∈ S`, and none of them meets `B(0, 1 - δ/2)`, while all of them sit inside
`B(0, 1 + δ/2)`.  Comparing volumes bounds `card S`.

The naive reading of that bound — `card ≲ δ^{-d}` — has one factor of `δ^{-1}`
too many: the centers lie on a sphere, not in a ball.  Keeping the inner ball
`B(0, 1 - δ/2)` in the ledger is what repairs it, since it turns the right-hand
side into the volume of a *shell*, `(1 + r)^d - (1 - r)^d ≍ 2 d r`; the sharp
form is read off in `Packing.Count`.

The crude corollary `card_le_of_separated` is enough to make a *maximal*
separated set exist, which is what the lower bound needs.
-/

import Grokking.Packing.Basic

open MeasureTheory Metric
open scoped ENNReal

namespace Grokking
namespace Packing

/-- **Packing ledger (upper).**  For a `δ`-separated set `S` of unit vectors,
the balls `B(x, δ/2)`, `x ∈ S`, and the ball `B(0, 1 - δ/2)` are pairwise
disjoint and all sit inside `B(0, 1 + δ/2)`; dividing by the volume of the unit
ball leaves

  `card S · (δ/2)^d + (1 - δ/2)^d ≤ (1 + δ/2)^d`.

Auxiliary (not from the paper): the volume form of the packing bound behind
the cardinality claim of §5. -/
theorem separated_volume_ledger (d : ℕ) (hd : 1 ≤ d) (δ : ℝ) (hδ : 0 < δ) (hδ2 : δ ≤ 2)
    (S : Finset (EucSpace d)) (hS : SeparatedOnSphere d S δ) :
    (S.card : ℝ) * (δ / 2) ^ d + (1 - δ / 2) ^ d ≤ (1 + δ / 2) ^ d := by
  set r := δ / 2 with hrdef
  have hr : 0 < r := by rw [hrdef]; linarith
  have hr1 : r ≤ 1 := by rw [hrdef]; linarith
  have hdisj : Set.PairwiseDisjoint (S : Set (EucSpace d)) (fun x => ball x r) := by
    intro x hx y hy hxy
    refine ball_disjoint_ball ?_
    have hsep := hS.2 x hx y hy hxy
    rw [dist_eq_norm]
    rw [hrdef]
    linarith
  have hunion : volume (⋃ x ∈ S, ball x r) = ∑ x ∈ S, volume (ball x r) :=
    measure_biUnion_finset hdisj fun _ _ => measurableSet_ball
  have hdisj0 : Disjoint (⋃ x ∈ S, ball x r) (closedBall (0 : EucSpace d) (1 - r)) := by
    rw [Set.disjoint_left]
    intro z hz hz0
    simp only [Set.mem_iUnion, exists_prop] at hz
    obtain ⟨x, hx, hzx⟩ := hz
    have h1 : ‖x‖ = 1 := hS.1 x hx
    have hxz : ‖x - z‖ < r := by
      rw [← norm_neg, neg_sub, ← dist_eq_norm]
      exact mem_ball.mp hzx
    have hz1 : ‖z‖ ≤ 1 - r := mem_closedBall_zero_iff.mp hz0
    have : ‖x‖ ≤ ‖x - z‖ + ‖z‖ := by simpa using norm_add_le (x - z) z
    linarith
  have hsub : (⋃ x ∈ S, ball x r) ∪ closedBall (0 : EucSpace d) (1 - r)
      ⊆ ball (0 : EucSpace d) (1 + r) := by
    intro z hz
    rcases (Set.mem_union _ _ _).mp hz with hz | hz
    · simp only [Set.mem_iUnion, exists_prop] at hz
      obtain ⟨x, hx, hzx⟩ := hz
      have h1 : ‖x‖ = 1 := hS.1 x hx
      have hzx' : ‖z - x‖ < r := by rw [← dist_eq_norm]; exact mem_ball.mp hzx
      have : ‖z‖ ≤ ‖z - x‖ + ‖x‖ := by simpa using norm_add_le (z - x) x
      rw [mem_ball_zero_iff]
      linarith
    · have : ‖z‖ ≤ 1 - r := mem_closedBall_zero_iff.mp hz
      rw [mem_ball_zero_iff]
      linarith
  have hle : volume (⋃ x ∈ S, ball x r) + volume (closedBall (0 : EucSpace d) (1 - r))
      ≤ volume (ball (0 : EucSpace d) (1 + r)) := by
    rw [← measure_union hdisj0 measurableSet_closedBall]
    exact measure_mono hsub
  rw [hunion, Finset.sum_congr rfl (fun x _ => volume_ball_eucSpace d hd x hr.le),
    Finset.sum_const, nsmul_eq_mul,
    volume_closedBall_eucSpace d hd _ (by linarith : (0 : ℝ) ≤ 1 - r),
    volume_ball_eucSpace d hd _ (by linarith : (0 : ℝ) ≤ 1 + r)] at hle
  have hle2 : ((S.card : ℝ≥0∞) * ENNReal.ofReal (r ^ d) + ENNReal.ofReal ((1 - r) ^ d))
      * volume (ball (0 : EucSpace d) 1)
      ≤ ENNReal.ofReal ((1 + r) ^ d) * volume (ball (0 : EucSpace d) 1) := by
    rw [add_mul, mul_assoc]
    exact hle
  have hcancel := (ENNReal.mul_le_mul_iff_left
    (measure_ball_pos volume (0 : EucSpace d) one_pos).ne' measure_ball_lt_top.ne).mp hle2
  rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (Nat.cast_nonneg _),
    ← ENNReal.ofReal_add (by positivity) (pow_nonneg (by linarith) d),
    ENNReal.ofReal_le_ofReal_iff (pow_nonneg (by linarith) d)] at hcancel
  exact hcancel

/-- The hypotheses of `separated_volume_ledger` are satisfiable: `d = 1`,
`δ = 1`, `S = ∅`. -/
example : SeparatedOnSphere 1 (∅ : Finset (EucSpace 1)) 1 :=
  ⟨fun _ hx => absurd hx (Finset.notMem_empty _), fun _ hx => absurd hx (Finset.notMem_empty _)⟩

/-- A crude cardinality bound in closed form: a `δ`-separated set of unit
vectors has at most `(4/δ)^d` elements.  Only the boundedness matters here —
it is what makes a *maximal* separated set exist; the sharp exponent
`(1/δ)^{d-1}` is `Packing.card_le_of_separatedOnSphere`.

Auxiliary (not from the paper). -/
theorem card_le_of_separated (d : ℕ) (hd : 1 ≤ d) (δ : ℝ) (hδ : 0 < δ) (hδ2 : δ ≤ 2)
    (S : Finset (EucSpace d)) (hS : SeparatedOnSphere d S δ) :
    (S.card : ℝ) ≤ (4 / δ) ^ d := by
  have hledger := separated_volume_ledger d hd δ hδ hδ2 S hS
  have hlow : (0 : ℝ) ≤ (1 - δ / 2) ^ d := pow_nonneg (by linarith) d
  have hhigh : (1 + δ / 2) ^ d ≤ 2 ^ d := pow_le_pow_left₀ (by linarith) (by linarith) d
  have hpos : (0 : ℝ) < (δ / 2) ^ d := by positivity
  have key : (S.card : ℝ) * (δ / 2) ^ d ≤ 2 ^ d := by linarith
  rw [show (4 / δ : ℝ) ^ d = 2 ^ d / (δ / 2) ^ d by
    rw [div_pow, div_pow, div_div_eq_mul_div, ← mul_pow]; norm_num, le_div_iff₀ hpos]
  exact key

/-- The hypotheses of `card_le_of_separated` are satisfiable: `d = 1`,
`δ = 1`, `S = ∅`. -/
example : SeparatedOnSphere 1 (∅ : Finset (EucSpace 1)) 1 :=
  ⟨fun _ hx => absurd hx (Finset.notMem_empty _), fun _ hx => absurd hx (Finset.notMem_empty _)⟩

/-- A maximal `δ`-separated set of unit vectors exists: the cardinalities of
`δ`-separated sets are bounded by `card_le_of_separated`, so one of largest
cardinality can be chosen, and nothing can be added to it.

Auxiliary (not from the paper). -/
theorem exists_maximalSeparated (d : ℕ) (hd : 1 ≤ d) (δ : ℝ) (hδ : 0 < δ) (hδ2 : δ ≤ 2) :
    ∃ S : Finset (EucSpace d), IsMaximalSeparated d S δ := by
  classical
  set P : ℕ → Prop := fun n => ∃ S : Finset (EucSpace d), SeparatedOnSphere d S δ ∧ S.card = n
    with hPdef
  set B : ℕ := ⌈(4 / δ) ^ d⌉₊ with hBdef
  have hPbound : ∀ n, P n → n ≤ B := by
    rintro n ⟨S, hS, rfl⟩
    have := card_le_of_separated d hd δ hδ hδ2 S hS
    exact Nat.cast_le.mp (le_trans this (Nat.le_ceil _))
  have hP0 : P 0 :=
    ⟨∅, ⟨fun _ hx => absurd hx (Finset.notMem_empty _),
      fun _ hx => absurd hx (Finset.notMem_empty _)⟩, rfl⟩
  obtain ⟨S, hS, hcard⟩ := Nat.findGreatest_spec (P := P) (Nat.zero_le B) hP0
  refine ⟨S, hS, ?_⟩
  by_contra hcov
  push Not at hcov
  obtain ⟨y, hy1, hy⟩ := hcov
  have hynot : y ∉ S := by
    intro hmem
    have hself := hy y hmem
    rw [sub_self, norm_zero] at hself
    linarith
  have hS' : SeparatedOnSphere d (insert y S) δ := by
    constructor
    · intro x hx
      rcases Finset.mem_insert.mp hx with rfl | hx
      · exact hy1
      · exact hS.1 x hx
    · intro a ha b hb hab
      rcases Finset.mem_insert.mp ha with rfl | ha'
      · rcases Finset.mem_insert.mp hb with rfl | hb'
        · exact absurd rfl hab
        · exact hy b hb'
      · rcases Finset.mem_insert.mp hb with rfl | hb'
        · rw [norm_sub_rev]; exact hy a ha'
        · exact hS.2 a ha' b hb' hab
  have hcard' : S.card + 1 ≤ Nat.findGreatest P B :=
    Nat.le_findGreatest (hPbound _ ⟨insert y S, hS', Finset.card_insert_of_notMem hynot⟩)
      ⟨insert y S, hS', Finset.card_insert_of_notMem hynot⟩
  omega

/-- The hypotheses of `exists_maximalSeparated` are satisfiable: `d = 1`,
`δ = 1`. -/
example : (1 : ℕ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 2 := ⟨le_rfl, one_pos, by norm_num⟩

end Packing
end Grokking
