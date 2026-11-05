# name: ca-holo-deriv-trig-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_cos_sin
# description: (cos z)' = -sin z, (sin z)' = cos z at 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'simp', True),
    (['1'], 'simp', False),
    (['Complex.I'], 'simp', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
