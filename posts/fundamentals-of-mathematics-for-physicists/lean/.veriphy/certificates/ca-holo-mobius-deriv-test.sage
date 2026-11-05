# name: ca-holo-mobius-deriv-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.mobius_deriv
# description: f'(z) = 2i/(z + i)^2 at 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '(by norm_num [Complex.ext_iff])'], "have hz := (by norm_num [Complex.ext_iff] : (0 : ℂ) + Complex.I ≠ 0)\nshow deriv (fun w : ℂ => (w - I) / (w + I)) (0 : ℂ) = 2 * I / ((0 : ℂ) + I) ^ 2\nhave h := ((hasDerivAt_id' (0 : ℂ)).sub_const I).div ((hasDerivAt_id' (0 : ℂ)).add_const I) hz\nrefine h.deriv.trans ?_\nring", True),
    (['1', '(by norm_num [Complex.ext_iff])'], "have hz := (by norm_num [Complex.ext_iff] : (1 : ℂ) + Complex.I ≠ 0)\nshow deriv (fun w : ℂ => (w - I) / (w + I)) (1 : ℂ) = 2 * I / ((1 : ℂ) + I) ^ 2\nhave h := ((hasDerivAt_id' (1 : ℂ)).sub_const I).div ((hasDerivAt_id' (1 : ℂ)).add_const I) hz\nrefine h.deriv.trans ?_\nring", False),
    (['Complex.I', '(by norm_num [Complex.ext_iff])'], "have hz := (by norm_num [Complex.ext_iff] : Complex.I + Complex.I ≠ 0)\nshow deriv (fun w : ℂ => (w - I) / (w + I)) (Complex.I) = 2 * I / ((Complex.I) + I) ^ 2\nhave h := ((hasDerivAt_id' (Complex.I)).sub_const I).div ((hasDerivAt_id' (Complex.I)).add_const I) hz\nrefine h.deriv.trans ?_\nring", False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
