# name: set-odd-diff-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Equivalence.odd_diff_not_equivalence
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '2', '3'], True, 'decide'),
    (['0', '5', '-2'], False, 'decide'),
    (['4', '4', '4'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
