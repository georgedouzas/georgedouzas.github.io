# name: ca-holo-closure-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.closure_ops
# description: closure under +, ×, ÷ for polynomial f, g on ℂ and on the unit disc; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '(fun _ : ℂ => (1 : ℂ))', 'Set.univ', '(by fun_prop)', '(by fun_prop)'], 'beta_reduce\nrefine ⟨by fun_prop, by fun_prop, fun h => ?_⟩\nexact DifferentiableOn.div (by fun_prop) (by fun_prop) h', True),
    (['(fun z : ℂ => z ^ 2)', '(fun z : ℂ => z + 3)', '(Metric.ball 0 1)', '(by fun_prop)', '(by fun_prop)'], 'beta_reduce\nrefine ⟨by fun_prop, by fun_prop, fun h => ?_⟩\nexact DifferentiableOn.div (by fun_prop) (by fun_prop) h', False),
    (['(fun z : ℂ => z * Complex.I)', '(fun z : ℂ => z ^ 2 + 1)', 'Set.univ', '(by fun_prop)', '(by fun_prop)'], 'beta_reduce\nrefine ⟨by fun_prop, by fun_prop, fun h => ?_⟩\nexact DifferentiableOn.div (by fun_prop) (by fun_prop) h', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
