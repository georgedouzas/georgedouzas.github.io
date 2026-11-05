# name: ra-sin-terms-one-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.sin_terms_one file "ra-sin-terms-one-test.sage"
# description: 1/(2m+1)! < 10⁻¹⁰ iff m ≥ 7, exactly (m = 0, 6, 7, 10)
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [Nat.factorial]'
CASES = [(['0'], True, None), (['6'], False, None), (['7'], False, None), (['10'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
