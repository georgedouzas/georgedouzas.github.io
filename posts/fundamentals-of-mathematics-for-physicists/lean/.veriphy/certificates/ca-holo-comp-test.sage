# name: ca-holo-comp-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.comp_holo
# description: composition of polynomials; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '(fun z : ℂ => z ^ 2)', 'Set.univ', 'Set.univ', '(by fun_prop)', '(by fun_prop)', '(Set.mapsTo_univ _ _)'], 'beta_reduce\nfun_prop', True),
    (['(fun z : ℂ => z + 1)', '(fun z : ℂ => z ^ 3)', 'Set.univ', 'Set.univ', '(by fun_prop)', '(by fun_prop)', '(Set.mapsTo_univ _ _)'], 'beta_reduce\nfun_prop', False),
    (['(fun z : ℂ => 2 * z)', '(fun z : ℂ => z * z + Complex.I)', '(Metric.ball 0 1)', 'Set.univ', '(by fun_prop)', '(by fun_prop)', '(Set.mapsTo_univ _ _)'], 'beta_reduce\nfun_prop', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
