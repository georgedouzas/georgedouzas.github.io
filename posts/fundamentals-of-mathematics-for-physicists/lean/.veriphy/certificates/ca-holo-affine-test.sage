# name: ca-holo-affine-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.affine_gcircle
# description: affine maps send generalised circles to generalised circles; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', '0', '(-1)', '1', '0', '0'], 'simp only [KaluzaKlein.ComplexAnalysis.ConformalMaps.gcircle, Complex.normSq_apply, map_add, map_mul, Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im,\n  Complex.conj_re, Complex.conj_im, Complex.ofReal_re, Complex.ofReal_im]\nring', True),
    (['0', '(1 + Complex.I)', '2', 'Complex.I', '1', '(2 - Complex.I)'], 'simp only [KaluzaKlein.ComplexAnalysis.ConformalMaps.gcircle, Complex.normSq_apply, map_add, map_mul, Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im,\n  Complex.conj_re, Complex.conj_im, Complex.ofReal_re, Complex.ofReal_im]\nring', False),
    (['2', 'Complex.I', '(-3)', '(1 + Complex.I)', '(-Complex.I)', '3'], 'simp only [KaluzaKlein.ComplexAnalysis.ConformalMaps.gcircle, Complex.normSq_apply, map_add, map_mul, Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im,\n  Complex.conj_re, Complex.conj_im, Complex.ofReal_re, Complex.ofReal_im]\nring', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
