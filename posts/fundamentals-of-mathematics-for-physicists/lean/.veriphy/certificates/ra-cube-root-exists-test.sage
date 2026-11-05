# name: ra-cube-root-exists-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.cube_root_exists file "ra-cube-root-exists-test.sage"
# description: sign change on [1, 2] and a rational cube root found by Sage (y = 27/8, 64/27, 125/64)
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['27 / 8', '(by norm_num)', '(by norm_num)'], False, 'exact ⟨by norm_num, by norm_num, 3 / 2, by norm_num, by norm_num⟩'), (['64 / 27', '(by norm_num)', '(by norm_num)'], False, 'exact ⟨by norm_num, by norm_num, 4 / 3, by norm_num, by norm_num⟩'), (['125 / 64', '(by norm_num)', '(by norm_num)'], True, 'exact ⟨by norm_num, by norm_num, 5 / 4, by norm_num, by norm_num⟩')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
