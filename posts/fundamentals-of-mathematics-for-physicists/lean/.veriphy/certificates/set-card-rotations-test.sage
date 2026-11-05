# name: set-card-rotations-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.card_rotations
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['6', 'inferInstance'], False, 'exact Complex.card_rootsOfUnity 6'),
    (['1', 'inferInstance'], True, 'exact Complex.card_rootsOfUnity 1'),
    (['4', 'inferInstance'], False, 'exact Complex.card_rootsOfUnity 4'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
