# name: ra-exp-remainder-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Series.exp_remainder file "ra-exp-remainder-test.sage"
# description: 1/(m+1)! ≤ e - s_m ≤ (m+2)/((m+1)!(m+1)) < 1/m!, with e bracketed by mathlib's 9-digit bounds
SEED = 0  # the instances are listed, no sampling
PROOF = 'have h1 := Real.exp_one_gt_d9\nhave h2 := Real.exp_one_lt_d9\nsimp only [KaluzaKlein.RealAnalysis.Series.expPartial, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]\nnorm_num\nrefine ⟨by linarith, by linarith⟩'
CASES = [(['1', '(by norm_num)'], True, None), (['3', '(by norm_num)'], False, None), (['6', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
