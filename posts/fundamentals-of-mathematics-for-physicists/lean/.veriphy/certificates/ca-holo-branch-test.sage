# name: ca-holo-branch-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.branch_hasDerivAt
# description: (Log z + 2πik)' = 1/z at (k, z) = (0, 1), (1, i), (-2, 2 + i); each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '1', '(by norm_num [Complex.mem_slitPlane_iff])'], 'exact (Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])).add_const _', True),
    (['1', 'Complex.I', '(by norm_num [Complex.mem_slitPlane_iff])'], 'exact (Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])).add_const _', False),
    (['(-2)', '(2 + Complex.I)', '(by norm_num [Complex.mem_slitPlane_iff])'], 'exact (Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])).add_const _', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
