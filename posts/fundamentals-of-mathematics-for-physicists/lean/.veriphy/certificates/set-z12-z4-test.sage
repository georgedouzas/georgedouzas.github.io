# name: set-z12-z4-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Products.z12_to_z4
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'decide'),
    (['1'], False, 'decide'),
    (['3'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
