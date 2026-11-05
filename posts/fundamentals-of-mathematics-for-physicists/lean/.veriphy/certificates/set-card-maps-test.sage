# name: set-card-maps-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Maps.card_maps
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3', '2'], False, 'decide'),
    (['0', '5'], True, 'decide'),
    (['2', '3'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
