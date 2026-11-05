# name: ra-euler-bound-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.euler_bound file "ra-euler-bound-test.sage"
# description: (1 + 1/n)^n < 3 in exact rational arithmetic
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num'
CASES = [(['1', '(by norm_num)'], True, None), (['2', '(by norm_num)'], False, None), (['10', '(by norm_num)'], False, None), (['50', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
