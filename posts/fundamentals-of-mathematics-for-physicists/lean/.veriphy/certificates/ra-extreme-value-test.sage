# name: ra-extreme-value-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.extreme_value file "ra-extreme-value-test.sage"
# description: the maximum and minimum points Sage finds (x² on [-1, 2], sin on [0, π], x³ - 2x on [0, 2])
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['(fun x : ℝ => x ^ 2)', '-1', '2', '(by norm_num)', '(by fun_prop)'], False, 'exact ⟨⟨2, by norm_num, fun x hx => by beta_reduce; nlinarith [hx.1, hx.2]⟩, ⟨0, by norm_num, fun x hx => by beta_reduce; nlinarith [sq_nonneg x]⟩⟩'), (['(fun x : ℝ => (1 : ℝ))', '0', '1', '(by norm_num)', '(by fun_prop)'], True, 'exact ⟨⟨0, by norm_num, fun x hx => le_rfl⟩, ⟨0, by norm_num, fun x hx => le_rfl⟩⟩'), (['(fun x : ℝ => x)', '-3', '5', '(by norm_num)', '(by fun_prop)'], False, 'exact ⟨⟨5, by norm_num, fun x hx => hx.2⟩, ⟨-3, by norm_num, fun x hx => hx.1⟩⟩')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
