# name: ra-alt-bounds-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.alt_bounds file "ra-alt-bounds-test.sage"
# description: (-1)^n + 1/n in (-1, 3/2], equal to 3/2 at n = 2
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num'
CASES = [(['1', '(by norm_num)'], True, None), (['2', '(by norm_num)'], False, None), (['3', '(by norm_num)'], False, None), (['7', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
