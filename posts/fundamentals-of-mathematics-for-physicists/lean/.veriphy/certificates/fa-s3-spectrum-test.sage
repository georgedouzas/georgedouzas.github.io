# name: fa-s3-spectrum-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.FourierAnalysis.PeterWeyl.s3_spectrum file "fa-s3-spectrum-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
PROOF = 'norm_num'
CASES = None
cases = CASES if CASES is not None else default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
