# name: set-class-mod-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Equivalence.class_mod
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3', '4', '0', '(by norm_num)'], False, 'decide'),
    (['3', '4', '1', '(by norm_num)'], False, 'decide'),
    (['3', '4', '2', '(by norm_num)'], False, 'decide'),
    (['1', '0', '0', '(by norm_num)'], True, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
