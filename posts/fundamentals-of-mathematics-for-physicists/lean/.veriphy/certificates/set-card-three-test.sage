# name: set-card-three-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.card_three
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℕ', 'inferInstance', '1', '2', '3', '(by decide)', '(by decide)', '(by decide)'], False, 'decide'),
    (['Fin 3', 'inferInstance', '0', '1', '2', '(by decide)', '(by decide)', '(by decide)'], True, 'decide'),
    (['ℤ', 'inferInstance', '-1', '0', '5', '(by decide)', '(by decide)', '(by decide)'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
