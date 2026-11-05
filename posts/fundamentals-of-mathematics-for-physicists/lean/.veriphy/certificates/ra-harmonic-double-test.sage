# name: ra-harmonic-double-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Series.harmonic_double file "ra-harmonic-double-test.sage"
# description: H_(2n) - H_n ≥ 1/2 exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [harmonic, Finset.sum_range_succ]'
CASES = [(['1', '(by norm_num)'], True, None), (['2', '(by norm_num)'], False, None), (['3', '(by norm_num)'], False, None), (['10', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
