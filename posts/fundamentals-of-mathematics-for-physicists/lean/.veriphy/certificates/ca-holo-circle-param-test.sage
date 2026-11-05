# name: ca-holo-circle-param-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.circle_param_deriv
# description: dz/dθ = i r e^{iθ} at sampled centres, radii, angles; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '0', '0'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.circle_param_deriv_aux 0 0 0', True),
    (['1', '2', '1'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.circle_param_deriv_aux 1 2 1', False),
    (['Complex.I', '(1 / 2)', '(-1)'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.circle_param_deriv_aux Complex.I (1 / 2) (-1)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
