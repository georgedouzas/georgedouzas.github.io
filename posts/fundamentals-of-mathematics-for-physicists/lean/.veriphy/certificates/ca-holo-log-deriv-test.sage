# name: ca-holo-log-deriv-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_hasDerivAt
# description: Log' z = 1/z at z = 1, i, 2 + i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', '(by norm_num [Complex.mem_slitPlane_iff])'], 'exact Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])', True),
    (['Complex.I', '(by norm_num [Complex.mem_slitPlane_iff])'], 'exact Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])', False),
    (['(2 + Complex.I)', '(by norm_num [Complex.mem_slitPlane_iff])'], 'exact Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
