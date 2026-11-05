# name: ra-sq-limit-delta-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.sq_limit_delta file "ra-sq-limit-delta-test.sage"
# description: |x - 2| < min(1, ε/5) ⇒ |x² - 4| < ε at sample ε, x
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [abs_lt]'
CASES = [(['1', '(by norm_num)', '2', '(by norm_num)'], True, None), (['1 / 10', '(by norm_num)', '201 / 100', '(by norm_num)'], False, None), (['5', '(by norm_num)', '23 / 10', '(by norm_num)'], False, None), (['1 / 2', '(by norm_num)', '99 / 50', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
