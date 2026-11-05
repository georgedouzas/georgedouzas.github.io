# name: ra-basel-partial-bound-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Series.basel_partial_bound file "ra-basel-partial-bound-test.sage"
# description: ∑_{k=1}^N 1/k² ≤ 2 - 1/N exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [baselTerm, Finset.sum_range_succ]'
CASES = [(['1', '(by norm_num)'], True, None), (['2', '(by norm_num)'], False, None), (['5', '(by norm_num)'], False, None), (['10', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
