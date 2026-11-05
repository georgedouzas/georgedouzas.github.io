# name: set-graph-components-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Connected.graph_components
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '2'], False, 'decide'),
    (['0', '3'], False, 'decide'),
    (['3', '4'], False, 'decide'),
    (['1', '1'], True, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
