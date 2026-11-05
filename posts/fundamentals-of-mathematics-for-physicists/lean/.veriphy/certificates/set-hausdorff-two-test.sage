# name: set-hausdorff-two-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.FiniteTopologies.hausdorff_two
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Finset.univ'], True, 'decide'),
    (['{∅, Finset.univ}'], False, 'decide'),
    (['{∅, {0}, Finset.univ}'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
