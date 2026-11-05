# name: set-card-zmod-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Quotients.card_zmod
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3', 'inferInstance'], False, 'decide'),
    (['5', 'inferInstance'], False, 'decide'),
    (['1', 'inferInstance'], True, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
