# name: fa-s9-value-of-bounds-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.FourierAnalysis.SeriesComputations.s9_value_of_bounds file "fa-s9-value-of-bounds-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
PROOF = 'norm_num'
CASES = [(['3.1415921', '(by norm_num)', '(by norm_num)'], True), (['3.1415925', '(by norm_num)', '(by norm_num)'], False), (['3.1415929', '(by norm_num)', '(by norm_num)'], True)]
cases = CASES if CASES is not None else default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
