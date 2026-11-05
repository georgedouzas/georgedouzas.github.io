# name: set-infinite-charzero-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.infinite_of_charZero
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℤ', 'inferInstance', 'inferInstance'], True, 'infer_instance'),
    (['ℝ', 'inferInstance', 'inferInstance'], False, 'exact Infinite.of_injective Nat.cast Nat.cast_injective'),
    (['ℂ', 'inferInstance', 'inferInstance'], False, 'exact Infinite.of_injective Nat.cast Nat.cast_injective'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
