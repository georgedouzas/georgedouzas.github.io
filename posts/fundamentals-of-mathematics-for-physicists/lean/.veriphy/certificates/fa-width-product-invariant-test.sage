# name: fa-width-product-invariant-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.FourierAnalysis.Uncertainty.width_product_invariant file "fa-width-product-invariant-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
PROOF = 'norm_num'
CASES = [(['1', '2', '1', 'one_ne_zero'], True), (['3/2', '-1', '2', 'two_ne_zero'], False), (['0', '5', '-3', '(by norm_num)'], False)]
cases = CASES if CASES is not None else default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
