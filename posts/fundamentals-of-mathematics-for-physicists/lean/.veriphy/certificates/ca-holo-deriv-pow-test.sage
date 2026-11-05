# name: ca-holo-deriv-pow-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow
# description: (z^n)' = n z^(n-1) at n = 0, 2, 3; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '0'], 'simp', True),
    (['2', '(1 + Complex.I)'], 'simp', False),
    (['3', 'Complex.I'], 'simp', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
