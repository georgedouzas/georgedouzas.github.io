# name: set-deriv-pos-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Homeomorphism.deriv_x_div_pos
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '(by constructor <;> norm_num)'], True, 'norm_num'),
    (['1/2', '(by constructor <;> norm_num)'], False, 'norm_num'),
    (['-9/10', '(by constructor <;> norm_num)'], False, 'norm_num'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
