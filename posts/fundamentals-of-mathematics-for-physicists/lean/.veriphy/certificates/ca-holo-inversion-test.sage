# name: ca-holo-inversion-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.inversion_gcircle
# description: 1/z maps generalised circles to generalised circles; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', '0', '(-1)', '1', 'one_ne_zero'], 'have hz := (one_ne_zero : (1 : ℂ) ≠ 0)\nhave hn : Complex.normSq (1 : ℂ) ≠ 0 := Complex.normSq_eq_zero.not.mpr hz\nsimp only [gcircle, map_inv₀, mul_re, conj_re, conj_im, Complex.inv_re, Complex.inv_im, Complex.normSq_conj]\nfield_simp\nring', True),
    (['0', '(1 + Complex.I)', '2', 'Complex.I', 'Complex.I_ne_zero'], 'have hz := (Complex.I_ne_zero)\nhave hn : Complex.normSq (Complex.I) ≠ 0 := Complex.normSq_eq_zero.not.mpr hz\nsimp only [gcircle, map_inv₀, mul_re, conj_re, conj_im, Complex.inv_re, Complex.inv_im, Complex.normSq_conj]\nfield_simp\nring', False),
    (['2', 'Complex.I', '(-3)', '(2 - Complex.I)', '(by norm_num [Complex.ext_iff])'], 'have hz := (by norm_num [Complex.ext_iff] : (2 - Complex.I : ℂ) ≠ 0)\nhave hn : Complex.normSq (2 - Complex.I) ≠ 0 := Complex.normSq_eq_zero.not.mpr hz\nsimp only [gcircle, map_inv₀, mul_re, conj_re, conj_im, Complex.inv_re, Complex.inv_im, Complex.normSq_conj]\nfield_simp\nring', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
