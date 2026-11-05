# name: set-doubles-subset-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.doubles_subset
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3'], False, 'decide'),
    (['0'], True, 'decide'),
    (['5'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
