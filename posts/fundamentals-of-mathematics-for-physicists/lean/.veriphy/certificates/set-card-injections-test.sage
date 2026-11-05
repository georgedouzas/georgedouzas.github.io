# name: set-card-injections-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Injective.card_injections
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3', '4'], False, 'rw [Fintype.card_embedding_eq]; decide'),
    (['0', '5'], True, 'rw [Fintype.card_embedding_eq]; decide'),
    (['2', '2'], False, 'rw [Fintype.card_embedding_eq]; decide'),
    (['3', '2'], False, 'rw [Fintype.card_embedding_eq]; decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
