# name: ra-id-no-extremum-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.id_no_extremum file "ra-id-no-extremum-test.sage"
# description: points of (0, 1) above and below each x
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['1 / 2', '(by norm_num)'], True, 'exact ⟨⟨3 / 4, by norm_num, by norm_num⟩, ⟨1 / 4, by norm_num, by norm_num⟩⟩'), (['1 / 100', '(by norm_num)'], False, 'exact ⟨⟨1 / 2, by norm_num, by norm_num⟩, ⟨1 / 200, by norm_num, by norm_num⟩⟩'), (['99 / 100', '(by norm_num)'], False, 'exact ⟨⟨199 / 200, by norm_num, by norm_num⟩, ⟨1 / 2, by norm_num, by norm_num⟩⟩')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
