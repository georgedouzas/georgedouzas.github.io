# name: ra-exppartial-tail-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.expPartial_tail file "ra-exppartial-tail-test.sage"
# description: |s_m - s_n| ≤ 2/(n+1)! for s_n = ∑_{k≤n} 1/k!, in exact rational arithmetic
SEED = 0  # the instances are listed, no sampling
PROOF = 'simp only [KaluzaKlein.RealAnalysis.Sequences.expPartial, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]\nnorm_num [abs_le]'
CASES = [(['0', '1', '(by norm_num)'], True, None), (['2', '5', '(by norm_num)'], False, None), (['4', '9', '(by norm_num)'], False, None), (['6', '20', '(by norm_num)'], False, None), (['8', '27', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
