# name: set-le-refl-trans-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Equivalence.le_refl_trans
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '2', '3'], True, 'decide'),
    (['4', '1', '2'], False, 'decide'),
    (['2', '2', '2'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
