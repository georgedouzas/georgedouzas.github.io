# name: set-disjoint-balls-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Hausdorff.disjoint_balls
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '0', '0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (x y : EuclideanSpace ℝ (Fin n)), Disjoint (ball x (dist x y / 2)) (ball y (dist x y / 2)) := by\n  intro n x y\n  exact (ball_disjoint_ball (by linarith))\nexact @key (0) (0) (0)'),
    (['2', '0', 'EuclideanSpace.single 0 1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (x y : EuclideanSpace ℝ (Fin n)), Disjoint (ball x (dist x y / 2)) (ball y (dist x y / 2)) := by\n  intro n x y\n  exact (ball_disjoint_ball (by linarith))\nexact @key (2) (0) (EuclideanSpace.single 0 1)'),
    (['3', 'EuclideanSpace.single 1 2', '0'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (x y : EuclideanSpace ℝ (Fin n)), Disjoint (ball x (dist x y / 2)) (ball y (dist x y / 2)) := by\n  intro n x y\n  exact (ball_disjoint_ball (by linarith))\nexact @key (3) (EuclideanSpace.single 1 2) (0)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
