# name: set-card-powerset-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.card_powerset
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 3', 'Finset.univ'], False, 'decide'),
    (['ℕ', '∅'], True, 'decide'),
    (['ℕ', '{1, 2, 3, 4}'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
