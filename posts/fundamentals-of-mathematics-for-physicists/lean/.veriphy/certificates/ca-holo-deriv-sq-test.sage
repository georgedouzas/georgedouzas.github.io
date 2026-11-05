# name: ca-holo-deriv-sq-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_sq
# description: (z^2)' = 2z at 0, 1 + i, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow]\nnorm_num', True),
    (['(1 + Complex.I)'], 'rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow]\nnorm_num', False),
    (['Complex.I'], 'rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow]\nnorm_num', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
