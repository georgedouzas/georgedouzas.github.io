# name: ca-holo-tangent-ne-zero-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.parabola_tangent_ne_zero
# description: the parabola tangents are non-zero off 0; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', '0', '(Or.inl one_ne_zero)'], 'norm_num', True),
    (['0', '2', '(Or.inr (by norm_num))'], 'norm_num', False),
    (['(-1)', '3', '(Or.inl (by norm_num))'], 'norm_num', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
