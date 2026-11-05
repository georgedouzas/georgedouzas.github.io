# name: set-surj-iff-inj-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Surjective.surjective_iff_injective_self
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3', '![1, 2, 0]'], False, 'decide'),
    (['0', '![]'], True, 'decide'),
    (['3', '![0, 0, 1]'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
