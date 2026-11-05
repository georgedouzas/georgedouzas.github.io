# name: ca-holo-re-quot-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_quot_axes
# description: Re(h)/h along the axes at t = 1, 2, -1/2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', '(by norm_num)'], 'norm_num', True),
    (['2', '(by norm_num)'], 'norm_num', False),
    (['(-1 / 2)', '(by norm_num)'], 'norm_num', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
