# name: set-card-prod-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Products.card_prod
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 2', 'Fin 3', 'inferInstance', 'inferInstance'], False, 'decide'),
    (['Fin 0', 'Fin 3', 'inferInstance', 'inferInstance'], True, 'decide'),
    (['Fin 2 × Fin 3', 'Fin 2', 'inferInstance', 'inferInstance'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
