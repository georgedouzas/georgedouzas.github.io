# name: set-sierpinski-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.FiniteTopologies.sierpinski_facts
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '0'], True, 'decide'),
    (['0', '1'], False, 'decide'),
    (['1', '0'], False, 'decide'),
    (['1', '1'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
