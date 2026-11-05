# name: ra-exp-terms-needed-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Series.exp_terms_needed file "ra-exp-terms-needed-test.sage"
# description: 1/m! < 10⁻¹⁰ iff m ≥ 14, exactly (m = 13, 14, 0, 20)
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [Nat.factorial]'
CASES = [(['0'], True, None), (['13'], False, None), (['14'], False, None), (['20'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
