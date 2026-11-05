# name: ra-euler-increasing-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.euler_increasing file "ra-euler-increasing-test.sage"
# description: (1 + 1/n)^n < (1 + 1/(n+1))^(n+1) in exact rational arithmetic
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num'
CASES = [(['1', '(by norm_num)'], True, None), (['2', '(by norm_num)'], False, None), (['3', '(by norm_num)'], False, None), (['10', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
