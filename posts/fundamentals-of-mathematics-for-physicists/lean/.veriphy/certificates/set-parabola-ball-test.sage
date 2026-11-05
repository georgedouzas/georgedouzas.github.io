# name: set-parabola-ball-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.OpenSets.parabola_ball
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '1', '0', '1', '(by norm_num)', '(by norm_num)', '(by norm_num)'], True, 'norm_num'),
    (['1', '3', '6/5', '14/5', '(by norm_num)', '(by norm_num)', '(by norm_num)'], False, 'norm_num'),
    (['-1', '2', '-11/10', '19/10', '(by norm_num)', '(by norm_num)', '(by norm_num)'], False, 'norm_num'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
