# name: set-card-fun-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Counting.card_fun
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 3', 'Fin 2', 'inferInstance', 'inferInstance', 'inferInstance'], False, 'decide'),
    (['Fin 0', 'Fin 5', 'inferInstance', 'inferInstance', 'inferInstance'], True, 'decide'),
    (['Fin 4', 'Fin 3', 'inferInstance', 'inferInstance', 'inferInstance'], False, 'simp'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
