# name: ra-sat-not-attained-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.sat_not_attained file "ra-sat-not-attained-test.sage"
# description: x/(1 + x) < 1 and a point above 1 - ε chosen by Sage
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['1', '(by norm_num)'], True, 'refine ⟨fun x hx => by rw [div_lt_one (by linarith)]; linarith, 1, by norm_num, by norm_num⟩'), (['1 / 10', '(by norm_num)'], False, 'refine ⟨fun x hx => by rw [div_lt_one (by linarith)]; linarith, 10, by norm_num, by norm_num⟩'), (['1 / 1000', '(by norm_num)'], False, 'refine ⟨fun x hx => by rw [div_lt_one (by linarith)]; linarith, 1000, by norm_num, by norm_num⟩')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
