import Grokking.Basic

/-!
The parameter-free XSA operator in src/gpt/reference.rs and src/gpt/gpu.rs.
The numerical norm floor prevents full orthogonality below the floor; the
orthogonality theorem explicitly retains the required lower norm bound.
-/

namespace Grokking

noncomputable def exclusive {d : ℕ} (floor : ℝ) (y v : EucSpace d) : EucSpace d :=
  let u := normalize floor 1 v
  y - inner (𝕜 := ℝ) y u • u

theorem exclusive_inner_direction {d : ℕ} (floor : ℝ) (hf : 0 < floor)
    (y v : EucSpace d) (hv : floor ≤ ‖v‖) :
    inner (𝕜 := ℝ) (exclusive floor y v) (normalize floor 1 v) = 0 := by
  have hu := normalize_norm floor 1 hf zero_le_one v hv
  simp [exclusive, inner_sub_left, inner_smul_left, hu]

theorem exclusive_inner_value {d : ℕ} (floor : ℝ) (hf : 0 < floor)
    (y v : EucSpace d) (hv : floor ≤ ‖v‖) : inner (𝕜 := ℝ) (exclusive floor y v) v = 0 := by
  have hn : ‖v‖ ≠ 0 := ne_of_gt (lt_of_lt_of_le hf hv)
  have h := exclusive_inner_direction floor hf y v hv
  rw [normalize, max_eq_left hv, inner_smul_right] at h
  simp only [one_div] at h
  exact (mul_eq_zero.mp h).resolve_left (inv_ne_zero hn)

theorem exclusive_zero_value {d : ℕ} (floor : ℝ) (y : EucSpace d) :
    exclusive floor y 0 = y := by simp [exclusive, normalize]

example : (1e-12 : ℝ) ≤ ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ := by norm_num

end Grokking
