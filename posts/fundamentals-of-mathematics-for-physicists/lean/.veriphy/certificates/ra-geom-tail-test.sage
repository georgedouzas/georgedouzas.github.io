# name: ra-geom-tail-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.geom_tail file "ra-geom-tail-test.sage"
# description: |a_m - a_n| ≤ |r|^(n+1)/(1 - |r|) for the geometric partial sums, exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'simp only [Finset.sum_range_succ, Finset.sum_range_zero]\nnorm_num [abs_le]'
CASES = [(['0', '(by norm_num)', '0', '1', '(by norm_num)'], True, None), (['1 / 2', '(by norm_num)', '1', '3', '(by norm_num)'], False, None), (['-1 / 3', '(by norm_num)', '2', '5', '(by norm_num)'], False, None), (['9 / 10', '(by norm_num)', '0', '4', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
