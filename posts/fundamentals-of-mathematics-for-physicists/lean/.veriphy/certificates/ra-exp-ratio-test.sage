# name: ra-exp-ratio-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Series.exp_ratio file "ra-exp-ratio-test.sage"
# description: the ratio |x|/(k + 1) of consecutive terms of ∑ x^k/k!, exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [Nat.factorial, abs_div]'
CASES = [(['1', '(by norm_num)', '0'], True, None), (['2', '(by norm_num)', '3'], False, None), (['-3', '(by norm_num)', '5'], False, None), (['1 / 2', '(by norm_num)', '1'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
