# name: ca-holo-deriv-zero-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyRiemann.deriv_zero_const
# description: functions with f' = 0: the constants 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun _ : ℂ => (0 : ℂ))', 'Set.univ', 'isOpen_univ', 'isPreconnected_univ', '(differentiableOn_const 0)', '(by simp)', '0', '(Set.mem_univ _)', '1', '(Set.mem_univ _)'], 'rfl', True),
    (['(fun _ : ℂ => (1 : ℂ))', '(Metric.ball 0 1)', 'Metric.isOpen_ball', '(convex_ball (0 : ℂ) 1).isPreconnected', '(differentiableOn_const 1)', '(by simp)', '0', '(Metric.mem_ball_self one_pos)', '(1 / 2)', '(by simp; norm_num)'], 'rfl', False),
    (['(fun _ : ℂ => Complex.I)', 'Set.univ', 'isOpen_univ', 'isPreconnected_univ', '(differentiableOn_const Complex.I)', '(by simp)', 'Complex.I', '(Set.mem_univ _)', '2', '(Set.mem_univ _)'], 'rfl', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
