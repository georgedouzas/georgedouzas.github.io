# name: ra-factorial-power-ratio-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Series.factorial_power_ratio file "ra-factorial-power-ratio-test.sage"
# description: ((n+1)!/(n+1)^(n+1))/(n!/n^n) = (n/(n+1))^n, exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [Nat.factorial]'
CASES = [(['1', '(by norm_num)'], True, None), (['2', '(by norm_num)'], False, None), (['3', '(by norm_num)'], False, None), (['6', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
