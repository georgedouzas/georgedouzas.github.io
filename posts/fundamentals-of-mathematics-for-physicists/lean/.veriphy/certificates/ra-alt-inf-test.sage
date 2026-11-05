# name: ra-alt-inf-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.alt_inf file "ra-alt-inf-test.sage"
# description: an odd n with (-1)^n + 1/n < -1 + ε: Sage picks n > 1/ε odd
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['1', '(by norm_num)'], True, 'exact ⟨3, by norm_num, by norm_num⟩'), (['1 / 10', '(by norm_num)'], False, 'exact ⟨11, by norm_num, by norm_num⟩'), (['1 / 1000', '(by norm_num)'], False, 'exact ⟨1001, by norm_num, by norm_num⟩')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
