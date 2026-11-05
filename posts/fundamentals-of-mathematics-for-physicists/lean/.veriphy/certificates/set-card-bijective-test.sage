# name: set-card-bijective-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Bijective.card_eq_of_bijective
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 3', 'Fin 3', 'inferInstance', 'inferInstance', 'id', 'Function.bijective_id'], True, 'decide'),
    (['Bool', 'Fin 2', 'inferInstance', 'inferInstance', 'fun b => if b then 1 else 0', '(by decide)'], False, 'decide'),
    (['Fin 3', 'Fin 3', 'inferInstance', 'inferInstance', '![2, 0, 1]', '(by decide)'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
