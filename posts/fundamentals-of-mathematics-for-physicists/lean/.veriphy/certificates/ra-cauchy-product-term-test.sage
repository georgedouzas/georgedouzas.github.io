# name: ra-cauchy-product-term-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Exercises.cauchy_product_term file "ra-cauchy-product-term-test.sage"
# description: the inner sum of the Cauchy product equals (x + y)ⁿ/n! exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'simp only [cauchyTerm, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]\nnorm_num'
CASES = [(['1', '2', '0'], True, None), (['1', '2', '3'], False, None), (['1 / 2', '-3', '4'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
