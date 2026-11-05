# name: ca-holo-add-mul-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.holo_add_mul
# description: sum and product rule on ℂ and the unit disc; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '(fun _ : ℂ => (1 : ℂ))', 'Set.univ', 'isOpen_univ', '(by fun_prop)', '(by fun_prop)'], 'beta_reduce\nrefine ⟨by fun_prop, by fun_prop, fun z _ => ?_⟩\nexact deriv_mul (by fun_prop) (by fun_prop)', True),
    (['(fun z : ℂ => z ^ 2)', '(fun z : ℂ => z + 3)', '(Metric.ball 0 1)', 'Metric.isOpen_ball', '(by fun_prop)', '(by fun_prop)'], 'beta_reduce\nrefine ⟨by fun_prop, by fun_prop, fun z _ => ?_⟩\nexact deriv_mul (by fun_prop) (by fun_prop)', False),
    (['(fun z : ℂ => z * Complex.I)', '(fun z : ℂ => z ^ 3)', 'Set.univ', 'isOpen_univ', '(by fun_prop)', '(by fun_prop)'], 'beta_reduce\nrefine ⟨by fun_prop, by fun_prop, fun z _ => ?_⟩\nexact deriv_mul (by fun_prop) (by fun_prop)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
