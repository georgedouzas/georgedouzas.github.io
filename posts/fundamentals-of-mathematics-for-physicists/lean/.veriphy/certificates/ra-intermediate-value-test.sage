# name: ra-intermediate-value-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.intermediate_value file "ra-intermediate-value-test.sage"
# description: the point c Sage solves for (x² = 1/4 on [0, 1], x = 3 on [0, 5], x³ = 8 on [1, 3])
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['(fun x : ℝ => x ^ 2)', '0', '1', '1 / 4', '(by norm_num)', '(by fun_prop)', '(by norm_num)'], False, 'exact ⟨1 / 2, by norm_num, by norm_num⟩'), (['(fun x : ℝ => x)', '0', '5', '3', '(by norm_num)', '(by fun_prop)', '(by norm_num)'], False, 'exact ⟨3, by norm_num, by norm_num⟩'), (['(fun x : ℝ => x ^ 3)', '1', '3', '8', '(by norm_num)', '(by fun_prop)', '(by norm_num)'], False, 'exact ⟨2, by norm_num, by norm_num⟩'), (['(fun x : ℝ => x)', '0', '0', '0', '(by norm_num)', '(by fun_prop)', '(by norm_num)'], True, 'exact ⟨0, by norm_num, by norm_num⟩')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
