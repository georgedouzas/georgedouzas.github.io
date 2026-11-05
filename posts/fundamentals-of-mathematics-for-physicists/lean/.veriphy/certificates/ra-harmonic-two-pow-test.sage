# name: ra-harmonic-two-pow-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Series.harmonic_two_pow file "ra-harmonic-two-pow-test.sage"
# description: H_(2^m) ≥ 1 + m/2 in exact rational arithmetic
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [harmonic, Finset.sum_range_succ]'
CASES = [(['0'], True, None), (['1'], False, None), (['2'], False, None), (['3'], False, None), (['5'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
