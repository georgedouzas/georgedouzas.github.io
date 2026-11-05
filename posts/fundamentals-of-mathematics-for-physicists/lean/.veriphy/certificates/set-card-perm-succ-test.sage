# name: set-card-perm-succ-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Counting.card_perm_succ
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['2'], False, 'decide'),
    (['0'], True, 'decide'),
    (['3'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
