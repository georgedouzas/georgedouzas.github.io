# name: set-cycle-minus-two-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Connected.cycle_minus_two
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '2', '(by decide)', '(by decide)'], False, 'decide'),
    (['1', '4', '(by decide)', '(by decide)'], False, 'decide'),
    (['5', '5', '(by decide)', '(by decide)'], True, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
