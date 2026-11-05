# name: ra-gaussian-domination-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_domination file "ra-gaussian-domination-test.sage"
# description: x² e^(-ax²) ≤ x² e^(-a₀x²) for a ≥ a₀, at sample values (exp monotone)
SEED = 0  # the instances are listed, no sampling
PROOF = 'rw [abs_of_nonneg (by positivity)]\nall_goals (apply mul_le_mul_of_nonneg_left _ (sq_nonneg _); all_goals (apply Real.exp_le_exp.2; norm_num))'
CASES = [(['1', '1', '0', '(by norm_num)'], True, None), (['2', '1', '1', '(by norm_num)'], False, None), (['3', '1 / 2', '-2', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
