# name: fa-lorentzian-half-width-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.FourierAnalysis.TransformComputations.lorentzian_half_width file "fa-lorentzian-half-width-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
PROOF = 'norm_num'
CASES = [(['1', '0', 'one_ne_zero'], True), (['2', '3', 'two_ne_zero'], False), (['-1/2', '5', '(by norm_num)'], False)]
cases = CASES if CASES is not None else default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
