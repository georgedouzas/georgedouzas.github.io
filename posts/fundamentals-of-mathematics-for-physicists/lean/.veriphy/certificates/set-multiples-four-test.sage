# name: set-multiples-four-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Products.multiples_of_four
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '(by decide)'], True, 'decide'),
    (['4', '(by decide)'], False, 'decide'),
    (['20', '(by decide)'], False, 'decide'),
    (['-4', '(by decide)'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
