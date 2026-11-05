# name: set-descfactorial-zero-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Counting.descFactorial_eq_zero
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3', '2', '(by norm_num)'], False, 'decide'),
    (['1', '0', '(by norm_num)'], True, 'decide'),
    (['4', '3', '(by norm_num)'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
