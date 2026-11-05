# name: set-complement-pair-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.TopologicalSpaces.complement_pair_isFinTop
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 3', 'inferInstance', 'inferInstance', '{0}'], False, 'decide'),
    (['Fin 3', 'inferInstance', 'inferInstance', '∅'], True, 'decide'),
    (['Fin 4', 'inferInstance', 'inferInstance', '{1, 2}'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
