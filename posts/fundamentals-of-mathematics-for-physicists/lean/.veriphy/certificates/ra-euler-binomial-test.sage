# name: ra-euler-binomial-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.euler_binomial file "ra-euler-binomial-test.sage"
# description: the binomial expansion of (1 + 1/n)^n, exactly at n = 1, 2, 3, 5
SEED = 0  # the instances are listed, no sampling
PROOF = 'simp only [eulerTerm, Finset.sum_range_succ, Finset.prod_range_succ, Finset.sum_range_zero, Finset.prod_range_zero, Nat.factorial]\nnorm_num'
CASES = [(['1', '(by norm_num)'], True, None), (['2', '(by norm_num)'], False, None), (['3', '(by norm_num)'], False, None), (['5', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
