# name: set-card-le-injective-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Counting.card_le_of_injective
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 2', 'Fin 3', 'inferInstance', 'inferInstance', 'Fin.castSucc', 'Fin.castSucc_injective 2'], False, 'decide'),
    (['Fin 0', 'Fin 0', 'inferInstance', 'inferInstance', 'id', 'Function.injective_id'], True, 'decide'),
    (['Fin 3', 'Fin 4', 'inferInstance', 'inferInstance', '![3, 2, 1]', '(by decide)'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
