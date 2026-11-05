# name: ra-close-to-limit-close-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.close_to_limit_close file "ra-close-to-limit-close-test.sage"
# description: the triangle-inequality step |a_m - a_n| < ε at sample values
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [abs_lt]'
CASES = [(['1', '1', '1', '1', '(by norm_num)', '(by norm_num)'], True, None), (['21 / 20', '19 / 20', '1', '1 / 5', '(by norm_num)', '(by norm_num)'], False, None), (['3', '-1 / 2', '1', '8', '(by norm_num)', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
