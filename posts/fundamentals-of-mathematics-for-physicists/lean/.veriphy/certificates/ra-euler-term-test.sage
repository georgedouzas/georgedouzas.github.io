# name: ra-euler-term-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.euler_term file "ra-euler-term-test.sage"
# description: (1 + 1/n)^n = (n+1)^n/n^n: 2, 9/4, 64/27, 625/256
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num'
CASES = [(['1', '(by norm_num)'], True, None), (['2', '(by norm_num)'], False, None), (['3', '(by norm_num)'], False, None), (['4', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
