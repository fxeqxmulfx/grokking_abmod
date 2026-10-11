import Mathlib

/-!
Real-valued specifications for geometric embeddings. The Rust implementation
uses f32, with independent f64 autodiff and numerical CUDA checks. These theorems
do not claim a verified compiler or exact floating-point transcendental functions.
-/

namespace Grokking

abbrev EucSpace (d : ℕ) := EuclideanSpace ℝ (Fin d)

instance instFactTwoPiPos : Fact (0 < 2 * Real.pi) := ⟨by positivity⟩
noncomputable abbrev Torus := AddCircle (2 * Real.pi)

/-- The direction of every nonzero effective embedding has unit norm. Adapted
from GPTMini.Bridge.SphereResidence, without its epsilon-free RMSNorm assumption. -/
theorem direction_norm {d : ℕ} (x : EucSpace d) (hx : x ≠ 0) :
    ‖‖x‖⁻¹ • x‖ = 1 := by
  rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hx)]

/-- The actual embedding-norm flag uses a positive floor and an explicit radius. -/
noncomputable def normalize {d : ℕ} (floor radius : ℝ) (x : EucSpace d) : EucSpace d :=
  (radius / max ‖x‖ floor) • x

/-- Above the floor, unit/rms embedding normalization has the requested radius. -/
theorem normalize_norm {d : ℕ} (floor radius : ℝ) (hf : 0 < floor) (hr : 0 ≤ radius)
    (x : EucSpace d) (hx : floor ≤ ‖x‖) : ‖normalize floor radius x‖ = radius := by
  have hn : ‖x‖ ≠ 0 := ne_of_gt (lt_of_lt_of_le hf hx)
  rw [normalize, max_eq_left hx, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (div_nonneg hr (norm_nonneg x)), div_mul_cancel₀ _ hn]

/-- Zero vectors remain zero; the floor avoids the singularity of normalization. -/
theorem normalize_zero (d : ℕ) (floor radius : ℝ) :
    normalize floor radius (0 : EucSpace d) = 0 := by simp [normalize]

/-- Parameter-free RMSNorm with the positive epsilon used by the model. -/
noncomputable def rmsNorm {d : ℕ} (epsilon : ℝ) (x : EucSpace d) : EucSpace d :=
  (Real.sqrt (‖x‖ ^ 2 / d + epsilon))⁻¹ • x

/-- RMSNorm with epsilon does not have exactly radius sqrt(d). This corrected
identity matches src/gpt/reference.rs instead of importing an epsilon-free claim. -/
theorem rmsNorm_norm_sq {d : ℕ} (hd : 0 < d) (epsilon : ℝ) (he : 0 < epsilon)
    (x : EucSpace d) :
    ‖rmsNorm epsilon x‖ ^ 2 = (d : ℝ) * ‖x‖ ^ 2 / (‖x‖ ^ 2 + d * epsilon) := by
  have hdR : (0 : ℝ) < d := by exact_mod_cast hd
  have hs : 0 < ‖x‖ ^ 2 / d + epsilon := by positivity
  rw [rmsNorm, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs, inv_pow,
    Real.sq_sqrt hs.le]
  field_simp

example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := ⟨zero_lt_one, zero_le_one⟩

end Grokking
