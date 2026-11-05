# name: ra-continuity-rules-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.continuity_rules file "ra-continuity-rules-test.sage"
# description: the rules at explicit continuous functions (sin, exp, x², cos)
SEED = 0  # the instances are listed, no sampling
PROOF = 'refine ⟨by fun_prop, by fun_prop, by fun_prop, fun h => by fun_prop⟩'
CASES = [(['Real.sin', 'Real.exp', 'Real.continuous_sin', 'Real.continuous_exp'], False, None), (['(fun x : ℝ => x ^ 2)', 'Real.cos', '(continuous_pow 2)', 'Real.continuous_cos'], False, None), (['(fun _ : ℝ => (0 : ℝ))', '(fun _ : ℝ => (1 : ℝ))', 'continuous_const', 'continuous_const'], True, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
