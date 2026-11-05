# name: ca-holo-real-axis-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_to_circle
# description: |f(x)| = 1 for the Cayley map at x = 0, 1, -3/2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'have h1 : ‖((0 : ℝ) : ℂ) - Complex.I‖ = ‖((0 : ℝ) : ℂ) + Complex.I‖ := by\n  rw [Complex.norm_def, Complex.norm_def, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (0)).1, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (0)).2]\nhave h2 : ((0 : ℝ) : ℂ) + Complex.I ≠ 0 := by\n  intro h\n  have := congrArg Complex.im h\n  simp at this\nrw [norm_div, h1, div_self (norm_ne_zero_iff.mpr h2)]', True),
    (['1'], 'have h1 : ‖((1 : ℝ) : ℂ) - Complex.I‖ = ‖((1 : ℝ) : ℂ) + Complex.I‖ := by\n  rw [Complex.norm_def, Complex.norm_def, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (1)).1, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (1)).2]\nhave h2 : ((1 : ℝ) : ℂ) + Complex.I ≠ 0 := by\n  intro h\n  have := congrArg Complex.im h\n  simp at this\nrw [norm_div, h1, div_self (norm_ne_zero_iff.mpr h2)]', False),
    (['(-3 / 2)'], 'have h1 : ‖((-3 / 2 : ℝ) : ℂ) - Complex.I‖ = ‖((-3 / 2 : ℝ) : ℂ) + Complex.I‖ := by\n  rw [Complex.norm_def, Complex.norm_def, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (-3 / 2)).1, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (-3 / 2)).2]\nhave h2 : ((-3 / 2 : ℝ) : ℂ) + Complex.I ≠ 0 := by\n  intro h\n  have := congrArg Complex.im h\n  simp at this\nrw [norm_div, h1, div_self (norm_ne_zero_iff.mpr h2)]', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
