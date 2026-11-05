# name: ca-holo-log-harmonic-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_norm_harmonic
# description: ln|z| harmonic at z = 1, i, -2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', '(by simp)'], 'exact analyticAt_id.harmonicAt_log_norm (by simp)', True),
    (['Complex.I', '(by simp)'], 'exact analyticAt_id.harmonicAt_log_norm (by simp)', False),
    (['(-2)', '(by norm_num)'], 'exact analyticAt_id.harmonicAt_log_norm (by norm_num)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
