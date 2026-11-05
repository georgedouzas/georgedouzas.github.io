# name: set-card-families-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.FiniteTopologies.card_families
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'decide'),
    (['1'], False, 'decide'),
    (['2'], False, 'decide'),
    (['3'], False, 'simp [Fintype.card_finset]'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
