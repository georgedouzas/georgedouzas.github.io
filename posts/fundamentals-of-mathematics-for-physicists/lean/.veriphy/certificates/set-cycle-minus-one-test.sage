# name: set-cycle-minus-one-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Connected.cycle_minus_one
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '4', '(by decide)', '(by decide)'], False, 'decide'),
    (['2', '2', '(by decide)', '(by decide)'], True, 'decide'),
    (['1', '3', '(by decide)', '(by decide)'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
