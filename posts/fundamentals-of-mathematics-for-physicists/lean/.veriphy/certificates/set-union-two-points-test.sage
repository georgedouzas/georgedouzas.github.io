# name: set-union-two-points-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.TopologicalSpaces.union_two_points
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['{∅, {0}, Finset.univ}', '{∅, {1}, Finset.univ}', '(by decide)', '(by decide)'], False, 'decide'),
    (['{∅, Finset.univ}', '{∅, Finset.univ}', '(by decide)', '(by decide)'], True, 'decide'),
    (['Finset.univ', '{∅, {0}, Finset.univ}', '(by decide)', '(by decide)'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
