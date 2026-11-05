# name: ca-holo-rn-derivs-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_derivs
# description: the r and θ derivatives of r^n cos nθ at (n, r, θ) = (0, 1, 0), (2, 2, 1), (-1, 1/2, -1); each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '1', '0', '(by norm_num)'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_derivs_aux 0 1 0 (by norm_num)', True),
    (['2', '2', '1', '(by norm_num)'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_derivs_aux 2 2 1 (by norm_num)', False),
    (['(-1)', '(1 / 2)', '(-1)', '(by norm_num)'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_derivs_aux (-1) (1 / 2) (-1) (by norm_num)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
