# name: ra-recip-eps-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.recip_eps file "ra-recip-eps-test.sage"
# description: for n > 1/ε, |1/n| < ε and |(-1)^n/n| < ε, at ε = 1, 1/10, 1/1000
SEED = 0  # the instances are listed, no sampling
PROOF = 'constructor <;> norm_num [abs_div]'
CASES = [(['1', '(by norm_num)', '2', '(by norm_num)'], True, None), (['1 / 10', '(by norm_num)', '11', '(by norm_num)'], False, None), (['1 / 1000', '(by norm_num)', '1001', '(by norm_num)'], False, None), (['1 / 10', '(by norm_num)', '12', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
