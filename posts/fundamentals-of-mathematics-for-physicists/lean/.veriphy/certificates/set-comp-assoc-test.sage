# name: set-comp-assoc-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Composition.comp_assoc
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 5', 'Fin 5', 'Fin 5', 'Fin 5', '![1, 2, 3, 4, 0]', '![0, 2, 1, 4, 3]', '![4, 3, 2, 1, 0]'], False, 'decide'),
    (['Fin 5', 'Fin 5', 'Fin 5', 'Fin 5', 'id', 'id', 'id'], True, 'decide'),
    (['Fin 5', 'Fin 5', 'Fin 5', 'Fin 5', '![2, 0, 4, 1, 3]', '![3, 4, 0, 2, 1]', '![1, 3, 0, 4, 2]'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
