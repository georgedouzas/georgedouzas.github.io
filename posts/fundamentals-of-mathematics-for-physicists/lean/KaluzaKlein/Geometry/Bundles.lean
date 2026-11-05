import Mathlib

/-!
# Bundles: displayed facts

A section of the Möbius band must vanish (the intermediate value theorem on a sign-changing function), the Hopf
map lands on the unit sphere, and it is constant along the circles `(a, b) ↦ (a e^{it}, b e^{it})`.
-/

namespace KaluzaKlein.Bundles

/-- A continuous function on `[0, 2π]` with `s(2π) = −s(0)` vanishes somewhere: a section of the Möbius band,
which changes sign across the glued edge, has a zero. -/
theorem mobius_section_zero (s : ℝ → ℝ) (hs : Continuous s) (h : s (2 * Real.pi) = -s 0) :
    ∃ x ∈ Set.Icc 0 (2 * Real.pi), s x = 0 := by
  have hpi : (0 : ℝ) ≤ 2 * Real.pi := by positivity
  rcases le_or_gt 0 (s 0) with h0 | h0
  · have : s (2 * Real.pi) ≤ 0 := by rw [h]; linarith
    exact intermediate_value_Icc' hpi hs.continuousOn ⟨this, h0⟩
  · have : 0 ≤ s (2 * Real.pi) := by rw [h]; linarith
    exact intermediate_value_Icc hpi hs.continuousOn ⟨h0.le, this⟩

/-- The Hopf map `(a, b) ↦ (2 Re(a b̄), 2 Im(a b̄), |a|² − |b|²)`. -/
noncomputable def hopf (a b : ℂ) : ℝ × ℝ × ℝ :=
  (2 * (a * (starRingEnd ℂ) b).re, 2 * (a * (starRingEnd ℂ) b).im, Complex.normSq a - Complex.normSq b)

/-- The Hopf map sends the unit three-sphere `|a|² + |b|² = 1` to the unit two-sphere. -/
theorem hopf_on_sphere (a b : ℂ) (h : Complex.normSq a + Complex.normSq b = 1) :
    (hopf a b).1 ^ 2 + (hopf a b).2.1 ^ 2 + (hopf a b).2.2 ^ 2 = 1 := by
  simp only [hopf, Complex.normSq_apply, Complex.mul_re, Complex.mul_im, Complex.conj_re, Complex.conj_im] at *
  nlinarith [h]

/-- The Hopf map is constant along the fibres: multiplying `a` and `b` by the same phase does not change it. -/
theorem hopf_phase (a b : ℂ) (t : ℝ) :
    hopf (a * Complex.exp (t * Complex.I)) (b * Complex.exp (t * Complex.I)) = hopf a b := by
  have hn : Complex.normSq (Complex.exp (t * Complex.I)) = 1 := by
    rw [Complex.normSq_eq_norm_sq, Complex.norm_exp_ofReal_mul_I]; norm_num
  simp only [hopf, map_mul, Complex.normSq_mul, hn, mul_one]
  have : a * Complex.exp (t * Complex.I) * ((starRingEnd ℂ) b * (starRingEnd ℂ) (Complex.exp (t * Complex.I))) =
      a * (starRingEnd ℂ) b * (Complex.exp (t * Complex.I) * (starRingEnd ℂ) (Complex.exp (t * Complex.I))) := by ring
  rw [this, Complex.mul_conj, hn]
  simp

end KaluzaKlein.Bundles
