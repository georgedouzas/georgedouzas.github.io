# name: fa-rp2-spectrum-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.FourierAnalysis.CosetModes.rp2_spectrum file "fa-rp2-spectrum-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
PROOF = 'decide'
CASES = [(['0', '(by decide)'], True), (['2', '(by decide)'], False), (['4', '(by decide)'], False)]
cases = CASES if CASES is not None else default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
