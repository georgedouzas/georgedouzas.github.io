# name: ca-holo-angle-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.mul_preserves_angle
# description: multiplication by a ≠ 0 preserves angles, a = 1, i, 2 + i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', 'one_ne_zero', '1', 'Complex.I'], 'have ha := (one_ne_zero : (1 : ℂ) ≠ 0)\nhave hm : (ContinuousLinearMap.mul ℂ ℂ (1 : ℂ)) ≠ 0 := by\n  intro h\n  have := congrArg (fun L : ℂ →L[ℂ] ℂ => L 1) h\n  simp at this <;> exact ha this\nhave hc := InnerProductGeometry.IsConformalMap.preserves_angle (isConformalMap_complex_linear hm) (1 : ℂ) (Complex.I)\nfirst | exact hc | simpa using hc', True),
    (['Complex.I', 'Complex.I_ne_zero', '1', '(1 + Complex.I)'], 'have ha := (Complex.I_ne_zero)\nhave hm : (ContinuousLinearMap.mul ℂ ℂ (Complex.I)) ≠ 0 := by\n  intro h\n  have := congrArg (fun L : ℂ →L[ℂ] ℂ => L 1) h\n  simp at this <;> exact ha this\nhave hc := InnerProductGeometry.IsConformalMap.preserves_angle (isConformalMap_complex_linear hm) (1 : ℂ) (1 + Complex.I)\nfirst | exact hc | simpa using hc', False),
    (['(2 + Complex.I)', '(by norm_num [Complex.ext_iff])', 'Complex.I', '3'], 'have ha := (by norm_num [Complex.ext_iff] : (2 + Complex.I : ℂ) ≠ 0)\nhave hm : (ContinuousLinearMap.mul ℂ ℂ (2 + Complex.I)) ≠ 0 := by\n  intro h\n  have := congrArg (fun L : ℂ →L[ℂ] ℂ => L 1) h\n  simp at this <;> exact ha this\nhave hc := InnerProductGeometry.IsConformalMap.preserves_angle (isConformalMap_complex_linear hm) (Complex.I) (3 : ℂ)\nfirst | exact hc | simpa using hc', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
