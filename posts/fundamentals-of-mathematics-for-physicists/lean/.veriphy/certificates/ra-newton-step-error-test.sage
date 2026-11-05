# name: ra-newton-step-error-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.newton_step_error file "ra-newton-step-error-test.sage"
# description: the error identity of a Newton step at rational a, c, s with s² = c
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num'
CASES = [(['1', '1', '1', '(by norm_num)', '(by norm_num)'], True, None), (['2', '4', '2', '(by norm_num)', '(by norm_num)'], False, None), (['3 / 2', '9 / 4', '-3 / 2', '(by norm_num)', '(by norm_num)'], False, None), (['5', '1', '1', '(by norm_num)', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
