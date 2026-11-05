# name: set-points-closed-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Hausdorff.points_closed
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ', 'inferInstance', 'inferInstance', '0'], True, 'exact isClosed_singleton'),
    (['ℝ × ℝ', 'inferInstance', 'inferInstance', '(1, 2)'], False, 'exact isClosed_singleton'),
    (['ℕ', 'inferInstance', 'inferInstance', '5'], False, 'exact isClosed_singleton'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
