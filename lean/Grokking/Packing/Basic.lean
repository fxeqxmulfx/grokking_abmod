/-
# Sphere packing — separated sets of unit vectors, and the volume of a ball

The R'enyi centers of §5 of 2411.04990v2 are, once the indexing is stripped
away, a `δ`-separated set of unit vectors, so their number is the packing
number of `𝕊^{d-1}` at scale `δ`.  This file carries the two predicates that
say so, and the one Mathlib fact the volume estimates of
`Packing.Upper` and `Packing.Lower` run on: a ball of
`EucSpace d` has volume `r^d` times the unit ball's.
-/

import Grokking.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

open MeasureTheory Metric

namespace Grokking
namespace Packing

/-- A finite `δ`-separated set of unit vectors: the values a R'enyi center
subsequence takes, stripped of the indexing.

Source: arXiv:2411.04990v2, §5 (`eq: renyi`, the defining property of the
R'enyi centers). -/
def SeparatedOnSphere (d : ℕ) (S : Finset (EucSpace d)) (δ : ℝ) : Prop :=
  (∀ x ∈ S, ‖x‖ = 1) ∧ ∀ x ∈ S, ∀ y ∈ S, x ≠ y → δ < ‖x - y‖

/-- `S` is a *maximal* `δ`-separated set of unit vectors: it is `δ`-separated
and nothing can be added to it, which for `δ > 0` is the same as saying that
every unit vector lies within `δ` of a point of `S`.

Source: arXiv:2411.04990v2, §5 — the R'enyi centers of a sequence that visits
the whole sphere are maximal in this sense, which is what makes their number a
lower bound and not only an upper one. -/
def IsMaximalSeparated (d : ℕ) (S : Finset (EucSpace d)) (δ : ℝ) : Prop :=
  SeparatedOnSphere d S δ ∧ ∀ y : EucSpace d, ‖y‖ = 1 → ∃ x ∈ S, ‖y - x‖ ≤ δ

/-- Auxiliary (not from the paper): the volume of a ball of `EucSpace d`, as a
multiple of the unit ball's, from `Measure.addHaar_ball`. -/
theorem volume_ball_eucSpace (d : ℕ) (hd : 1 ≤ d) (x : EucSpace d) {r : ℝ} (hr : 0 ≤ r) :
    volume (ball x r) = ENNReal.ofReal (r ^ d) * volume (ball (0 : EucSpace d) 1) := by
  have : Nontrivial (EucSpace d) :=
    Module.nontrivial_of_finrank_pos (R := ℝ) (by rw [finrank_euclideanSpace_fin]; omega)
  rw [Measure.addHaar_ball volume x hr, finrank_euclideanSpace_fin]

example : (1 : ℕ) ≤ 1 ∧ (0 : ℝ) ≤ 1 := ⟨le_rfl, zero_le_one⟩

/-- Auxiliary (not from the paper): the same for a closed ball. -/
theorem volume_closedBall_eucSpace (d : ℕ) (hd : 1 ≤ d) (x : EucSpace d) {r : ℝ} (hr : 0 ≤ r) :
    volume (closedBall x r) = ENNReal.ofReal (r ^ d) * volume (ball (0 : EucSpace d) 1) := by
  have : Nontrivial (EucSpace d) :=
    Module.nontrivial_of_finrank_pos (R := ℝ) (by rw [finrank_euclideanSpace_fin]; omega)
  rw [Measure.addHaar_closedBall volume x hr, finrank_euclideanSpace_fin]

end Packing
end Grokking
