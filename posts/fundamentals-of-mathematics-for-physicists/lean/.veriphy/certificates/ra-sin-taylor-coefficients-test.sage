# name: ra-sin-taylor-coefficients-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.sin_taylor_coefficients file "ra-sin-taylor-coefficients-test.sage"
# description: the coefficients 1, -1/6, 1/120, -1/5040 exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [Nat.factorial]'
CASES = [(['0'], True, None), (['1'], False, None), (['2'], False, None), (['3'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
