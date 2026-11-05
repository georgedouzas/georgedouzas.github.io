# name: ra-x-sq-sums-rate-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.x_sq_sums_rate file "ra-x-sq-sums-rate-test.sage"
# description: the closed forms equal 1/3 ∓ 1/(2n) + 1/(6n²) exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num'
CASES = [(['1', '(by norm_num)'], True, None), (['4', '(by norm_num)'], False, None), (['16', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
