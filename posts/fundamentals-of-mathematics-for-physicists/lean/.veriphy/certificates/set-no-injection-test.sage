# name: set-no-injection-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Injective.no_injection
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3', '2', '(by norm_num)', '![0, 1, 1]'], False, 'decide'),
    (['2', '1', '(by norm_num)', '![0, 0]'], True, 'decide'),
    (['4', '3', '(by norm_num)', '![0, 1, 2, 0]'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
