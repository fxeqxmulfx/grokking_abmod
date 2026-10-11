/-
# RoPE — Rotary Positional Encoding

Formalization of `rope_tables` and `apply_rope` from `reference/model.py`:

```python
def rope_tables(head_dim, max_seq_len, theta):
    half   = head_dim // 2
    inv_f  = theta ** (-arange(half) / half)
    pos    = arange(max_seq_len)
    angles = outer(pos, inv_f)              # (T, half)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

RoPE rotates the pair `(x_k, x_{k + half})` by the angle `t · invFreq k`, the
two halves of the head dimension being paired by `chunk(2)`.  `ropeSplit` is
that pairing: it presents `Fin d_head` as two copies of `Fin (d_head / 2)`
followed by the coordinate an odd `d_head` leaves over, which RoPE does not
touch.  Everything else is stated and proved on the split index, where the
rotation is two-dimensional and the sums factor.

Proved here:
  - `applyRope_isometry` — RoPE preserves the L2 norm,
  - `applyRope_relative` — inner products depend on the difference of the
    positions only, which is what makes RoPE a *relative* encoding.
-/

import Grokking.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2

open scoped BigOperators
open Real

namespace Grokking
namespace RoPE

/-- Inverse-frequency `inv_freq k = theta^{-2k/d_head}` for `k = 0,…,d_head/2-1`. -/
noncomputable def invFreq (d_head : ℕ) (theta : ℝ) (k : Fin (d_head / 2)) : ℝ :=
  theta ^ (-((2 * (k : ℕ) : ℕ) : ℝ) / (d_head : ℝ))

/-- The pairing of coordinates that `chunk(2, dim=-1)` performs: the head
dimension is two copies of `Fin (d_head / 2)` — the halves `x1` and `x2` —
followed by the leftover coordinate of an odd `d_head`. -/
def ropeSplit (d_head : ℕ) :
    Fin d_head ≃ (Fin (d_head / 2) ⊕ Fin (d_head / 2)) ⊕ Fin (d_head - 2 * (d_head / 2)) :=
  (finCongr (by omega : d_head = d_head / 2 + d_head / 2 + (d_head - 2 * (d_head / 2)))).trans
    (finSumFinEquiv.symm.trans (Equiv.sumCongr finSumFinEquiv.symm (Equiv.refl _)))

/-- The `x1`-coordinate of the rotation pair `k`. -/
def ropeFst (d_head : ℕ) (k : Fin (d_head / 2)) : Fin d_head :=
  (ropeSplit d_head).symm (Sum.inl (Sum.inl k))

/-- The `x2`-coordinate of the rotation pair `k`. -/
def ropeSnd (d_head : ℕ) (k : Fin (d_head / 2)) : Fin d_head :=
  (ropeSplit d_head).symm (Sum.inl (Sum.inr k))

@[simp] theorem ropeSplit_symm_inl_inl (d_head : ℕ) (k : Fin (d_head / 2)) :
    (ropeSplit d_head).symm (Sum.inl (Sum.inl k)) = ropeFst d_head k := rfl

@[simp] theorem ropeSplit_symm_inl_inr (d_head : ℕ) (k : Fin (d_head / 2)) :
    (ropeSplit d_head).symm (Sum.inl (Sum.inr k)) = ropeSnd d_head k := rfl

/-- The angle by which the pair `k` is rotated at position `t`:
`t · invFreq k`, the `angles` table of `rope_tables`. -/
noncomputable def ropeAngle (d_head : ℕ) (theta t : ℝ) (k : Fin (d_head / 2)) : ℝ :=
  t * invFreq d_head theta k

/-- RoPE on the split index: the pair `k` is rotated by `ropeAngle k`, and the
leftover coordinate of an odd `d_head` is left alone. -/
noncomputable def ropeCoord
    (d_head : ℕ) (theta t : ℝ) (x : EucSpace d_head) :
    (Fin (d_head / 2) ⊕ Fin (d_head / 2)) ⊕ Fin (d_head - 2 * (d_head / 2)) → ℝ :=
  Sum.elim
    (Sum.elim
      (fun k => x (ropeFst d_head k) * Real.cos (ropeAngle d_head theta t k)
        - x (ropeSnd d_head k) * Real.sin (ropeAngle d_head theta t k))
      (fun k => x (ropeFst d_head k) * Real.sin (ropeAngle d_head theta t k)
        + x (ropeSnd d_head k) * Real.cos (ropeAngle d_head theta t k)))
    (fun j => x ((ropeSplit d_head).symm (Sum.inr j)))

/-- RoPE rotation applied to a head-dim vector at position `t`:

  `x1' = x1 * cos(t · invFreq) - x2 * sin(t · invFreq)`,
  `x2' = x1 * sin(t · invFreq) + x2 * cos(t · invFreq)`. -/
noncomputable def applyRope
    (d_head : ℕ) (theta : ℝ) (t : ℝ) (x : EucSpace d_head) : EucSpace d_head :=
  (EuclideanSpace.equiv (Fin d_head) ℝ).symm
    (fun i => ropeCoord d_head theta t x (ropeSplit d_head i))

@[simp] theorem applyRope_apply
    (d_head : ℕ) (theta t : ℝ) (x : EucSpace d_head) (i : Fin d_head) :
    applyRope d_head theta t x i = ropeCoord d_head theta t x (ropeSplit d_head i) := rfl

/-- The sum of any function of the coordinates, read on the split index. -/
theorem sum_split (d_head : ℕ) (f : Fin d_head → ℝ) :
    ∑ i : Fin d_head, f i
      = (∑ k : Fin (d_head / 2), f (ropeFst d_head k)
          + ∑ k : Fin (d_head / 2), f (ropeSnd d_head k))
        + ∑ j : Fin (d_head - 2 * (d_head / 2)), f ((ropeSplit d_head).symm (Sum.inr j)) := by
  rw [← Equiv.sum_comp (ropeSplit d_head).symm f, Fintype.sum_sum_type, Fintype.sum_sum_type]
  rfl

/-- The same sum for a function of the split index. -/
theorem sum_split' (d_head : ℕ)
    (g : (Fin (d_head / 2) ⊕ Fin (d_head / 2)) ⊕ Fin (d_head - 2 * (d_head / 2)) → ℝ) :
    ∑ i : Fin d_head, g (ropeSplit d_head i)
      = (∑ k : Fin (d_head / 2), g (Sum.inl (Sum.inl k))
          + ∑ k : Fin (d_head / 2), g (Sum.inl (Sum.inr k)))
        + ∑ j : Fin (d_head - 2 * (d_head / 2)), g (Sum.inr j) := by
  rw [Equiv.sum_comp (ropeSplit d_head) g, Fintype.sum_sum_type, Fintype.sum_sum_type]

/-- **Isometry of RoPE.**

`apply_rope` is an L2-isometry: `‖applyRope d theta t x‖ = ‖x‖`, because each
pair is turned by a plane rotation and the leftover coordinate is untouched.
Source: `reference/model.py` (`apply_rope`). -/
theorem applyRope_isometry
    (d_head : ℕ) (theta : ℝ) (t : ℝ) (x : EucSpace d_head) :
    ‖applyRope d_head theta t x‖ = ‖x‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  congr 1
  rw [show (fun i : Fin d_head => ‖applyRope d_head theta t x i‖ ^ 2)
      = (fun i : Fin d_head => ‖ropeCoord d_head theta t x (ropeSplit d_head i)‖ ^ 2) from rfl]
  rw [sum_split' d_head (fun j => ‖ropeCoord d_head theta t x j‖ ^ 2),
    sum_split d_head (fun i => ‖x i‖ ^ 2)]
  congr 1
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [ropeCoord, Sum.elim_inl, Sum.elim_inr, Real.norm_eq_abs, sq_abs]
  linear_combination (x (ropeFst d_head k) ^ 2 + x (ropeSnd d_head k) ^ 2)
    * Real.sin_sq_add_cos_sq (ropeAngle d_head theta t k)

/-- RoPE acts linearly on the coordinates, so it commutes with differences. -/
theorem ropeCoord_sub (d_head : ℕ) (theta t : ℝ) (x y : EucSpace d_head)
    (s : (Fin (d_head / 2) ⊕ Fin (d_head / 2)) ⊕ Fin (d_head - 2 * (d_head / 2))) :
    ropeCoord d_head theta t (x - y) s
      = ropeCoord d_head theta t x s - ropeCoord d_head theta t y s := by
  rcases s with (k | k) | j <;> simp [ropeCoord] <;> ring

/-- The rotation of a difference is the difference of the rotations. -/
theorem applyRope_sub (d_head : ℕ) (theta t : ℝ) (x y : EucSpace d_head) :
    applyRope d_head theta t (x - y)
      = applyRope d_head theta t x - applyRope d_head theta t y := by
  ext i
  simp [applyRope_apply, ropeCoord_sub]

/-- **RoPE preserves distances**, being linear and norm-preserving.  This is
what lets a Lipschitz estimate for the head ignore the positional encoding
entirely.  Source: `reference/model.py` (`apply_rope`). -/
theorem applyRope_dist (d_head : ℕ) (theta t : ℝ) (x y : EucSpace d_head) :
    ‖applyRope d_head theta t x - applyRope d_head theta t y‖ = ‖x - y‖ := by
  rw [← applyRope_sub, applyRope_isometry]

/-- The angle at position `s - t` is the difference of the angles: RoPE is a
one-parameter group of rotations, which is what `applyRope_relative` rests on. -/
theorem ropeAngle_sub (d_head : ℕ) (theta t s : ℝ) (k : Fin (d_head / 2)) :
    ropeAngle d_head theta (s - t) k
      = ropeAngle d_head theta s k - ropeAngle d_head theta t k := by
  unfold ropeAngle; ring

/-- **Relative positional structure.**

  `⟨applyRope d θ t x, applyRope d θ s y⟩ = ⟨x, applyRope d θ (s - t) y⟩`:

inner products depend only on the *difference* of the positions, which is the
property RoPE attention exploits.  Source: `reference/model.py`
(`apply_rope`); Su et al., *RoFormer*, eq. (14). -/
theorem applyRope_relative
    (d_head : ℕ) (theta : ℝ) (t s : ℝ) (x y : EucSpace d_head) :
    inner (𝕜 := ℝ) (applyRope d_head theta t x) (applyRope d_head theta s y)
      = inner (𝕜 := ℝ) x (applyRope d_head theta (s - t) y) := by
  have hL : ∀ i : Fin d_head,
      applyRope d_head theta s y i * applyRope d_head theta t x i
        = (fun j => ropeCoord d_head theta s y j * ropeCoord d_head theta t x j)
            (ropeSplit d_head i) := fun _ => rfl
  have hR : ∀ i : Fin d_head,
      applyRope d_head theta (s - t) y i * x i
        = (fun j => ropeCoord d_head theta (s - t) y j * x ((ropeSplit d_head).symm j))
            (ropeSplit d_head i) := by
    intro i; simp [applyRope]
  rw [PiLp.inner_apply, PiLp.inner_apply]
  simp only [RCLike.inner_apply, conj_trivial]
  rw [Finset.sum_congr rfl (fun i _ => hL i), Finset.sum_congr rfl (fun i _ => hR i),
    sum_split' d_head (fun j => ropeCoord d_head theta s y j * ropeCoord d_head theta t x j),
    sum_split' d_head
      (fun j => ropeCoord d_head theta (s - t) y j * x ((ropeSplit d_head).symm j))]
  congr 1
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [ropeCoord, Sum.elim_inl, Sum.elim_inr, ropeSplit_symm_inl_inl,
    ropeSplit_symm_inl_inr, ropeAngle_sub, Real.cos_sub, Real.sin_sub]
  ring

end RoPE
end Grokking
