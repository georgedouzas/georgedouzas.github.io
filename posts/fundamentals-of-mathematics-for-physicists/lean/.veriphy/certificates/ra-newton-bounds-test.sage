# name: ra-newton-bounds-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.newton_bounds file "ra-newton-bounds-test.sage"
# description: the Newton iterates for √c are positive, satisfy a_n² ≥ c and decrease, computed exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [KaluzaKlein.RealAnalysis.Sequences.newton]'
CASES = [(['1', '(by norm_num)', '0'], True, None), (['2', '(by norm_num)', '0'], False, None), (['2', '(by norm_num)', '2'], False, None), (['3', '(by norm_num)', '1'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
