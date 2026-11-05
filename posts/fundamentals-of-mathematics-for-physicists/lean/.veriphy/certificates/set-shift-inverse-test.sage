# name: set-shift-inverse-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Inverse.shift_inverse
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['6', '1'], False, 'decide'),
    (['1', '0'], True, 'decide'),
    (['5', '3'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
