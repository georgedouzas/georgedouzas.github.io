# name: ra-rat-fun-eps-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.rat_fun_eps file "ra-rat-fun-eps-test.sage"
# description: |(2n + 1)/(n + 3) - 2| = 5/(n + 3) < ε for n > 5/ε - 3
SEED = 0  # the instances are listed, no sampling
PROOF = 'constructor <;> norm_num [abs_div]'
CASES = [(['1', '(by norm_num)', '5', '(by norm_num)'], True, None), (['1 / 10', '(by norm_num)', '48', '(by norm_num)'], False, None), (['1 / 100', '(by norm_num)', '498', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
