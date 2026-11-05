# name: set-ball-open-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.OpenSets.ball_radius_subset
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '0', '0', '1', '(by simp)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (p x : EuclideanSpace ℝ (Fin n)) (ε : ℝ) (hx : x ∈ ball p ε), 0 < ε - dist x p ∧ ball x (ε - dist x p) ⊆ ball p ε ∧ IsOpen (ball p ε) := by\n  intro n p x ε hx\n  refine ⟨sub_pos.2 hx, fun y hy => ?_, isOpen_ball⟩\n  rw [mem_ball] at hy ⊢\n  linarith [dist_triangle y x p]\nexact @key (0) (0) (0) (1) ((by simp))'),
    (['1', '0', '0', '2', '(by simp)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (p x : EuclideanSpace ℝ (Fin n)) (ε : ℝ) (hx : x ∈ ball p ε), 0 < ε - dist x p ∧ ball x (ε - dist x p) ⊆ ball p ε ∧ IsOpen (ball p ε) := by\n  intro n p x ε hx\n  refine ⟨sub_pos.2 hx, fun y hy => ?_, isOpen_ball⟩\n  rw [mem_ball] at hy ⊢\n  linarith [dist_triangle y x p]\nexact @key (1) (0) (0) (2) ((by simp))'),
    (['3', '0', '0', '1/2', '(by simp)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (p x : EuclideanSpace ℝ (Fin n)) (ε : ℝ) (hx : x ∈ ball p ε), 0 < ε - dist x p ∧ ball x (ε - dist x p) ⊆ ball p ε ∧ IsOpen (ball p ε) := by\n  intro n p x ε hx\n  refine ⟨sub_pos.2 hx, fun y hy => ?_, isOpen_ball⟩\n  rw [mem_ball] at hy ⊢\n  linarith [dist_triangle y x p]\nexact @key (3) (0) (0) (1/2) ((by simp))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
