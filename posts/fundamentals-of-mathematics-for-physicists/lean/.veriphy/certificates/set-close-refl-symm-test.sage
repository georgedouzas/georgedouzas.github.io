# name: set-close-refl-symm-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Equivalence.close_refl_symm
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '2'], True, 'decide'),
    (['1', '3'], False, 'decide'),
    (['4', '4'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
