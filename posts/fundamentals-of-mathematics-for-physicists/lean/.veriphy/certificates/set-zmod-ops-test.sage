# name: set-zmod-ops-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Quotients.zmod_ops
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3', 'inferInstance', '1', '2'], False, 'decide'),
    (['3', 'inferInstance', '2', '2'], False, 'decide'),
    (['5', 'inferInstance', '2', '3'], False, 'decide'),
    (['5', 'inferInstance', '4', '4'], False, 'decide'),
    (['1', 'inferInstance', '0', '0'], True, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
