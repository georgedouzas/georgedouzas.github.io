# name: ca-holo-no-global-log-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_global_log_conjugate
# description: ln|z - c| has no conjugate on C minus {c}, c = 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_global_log_conjugate_aux 0', True),
    (['1'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_global_log_conjugate_aux 1', False),
    (['Complex.I'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_global_log_conjugate_aux Complex.I', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
