# name: ca-holo-const-norm-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.const_of_norm_const
# description: constant-modulus holomorphic functions 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun _ : ℂ => (0 : ℂ))', 'Set.univ', 'isOpen_univ', 'isPreconnected_univ', '(differentiableOn_const 0)', '0', '(by simp)'], 'exact ⟨_, fun _ _ => rfl⟩', True),
    (['(fun _ : ℂ => (1 : ℂ))', '(Metric.ball 0 1)', 'Metric.isOpen_ball', '(convex_ball (0 : ℂ) 1).isPreconnected', '(differentiableOn_const 1)', '1', '(by simp)'], 'exact ⟨_, fun _ _ => rfl⟩', False),
    (['(fun _ : ℂ => Complex.I)', 'Set.univ', 'isOpen_univ', 'isPreconnected_univ', '(differentiableOn_const Complex.I)', '1', '(by simp)'], 'exact ⟨_, fun _ _ => rfl⟩', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
