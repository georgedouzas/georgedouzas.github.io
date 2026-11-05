# name: set-between-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.FiniteTopologies.between_indiscrete_discrete
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 3', 'inferInstance', 'inferInstance', '{∅, {0}, {0, 1}, Finset.univ}', '(by decide)'], False, 'decide'),
    (['Fin 0', 'inferInstance', 'inferInstance', '{∅}', '(by decide)'], True, 'decide'),
    (['Fin 2', 'inferInstance', 'inferInstance', 'Finset.univ', '(by decide)'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
