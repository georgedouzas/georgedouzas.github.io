# name: set-rotations-noncommute-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Composition.rotations_noncommute
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Real.pi / 2', 'Real.pi / 2', '(by simp)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (α β : ℝ) (h : Real.sin α * Real.sin β ≠ 0), Rz α * Rx β ≠ Rx β * Rz α := by\n  intro α β h\n  intro hc\n  have := congrFun (congrFun hc 0) 2\n  rw [Rz_mul_Rx, Rx_mul_Rz] at this\n  simp at this\n  exact h (mul_eq_zero.2 this)\nexact @key (Real.pi / 2) (Real.pi / 2) ((by simp))'),
    (['Real.pi / 2', '-(Real.pi / 2)', '(by simp)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (α β : ℝ) (h : Real.sin α * Real.sin β ≠ 0), Rz α * Rx β ≠ Rx β * Rz α := by\n  intro α β h\n  intro hc\n  have := congrFun (congrFun hc 0) 2\n  rw [Rz_mul_Rx, Rx_mul_Rz] at this\n  simp at this\n  exact h (mul_eq_zero.2 this)\nexact @key (Real.pi / 2) (-(Real.pi / 2)) ((by simp))'),
    (['Real.pi / 6', 'Real.pi / 2', '(by simp)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (α β : ℝ) (h : Real.sin α * Real.sin β ≠ 0), Rz α * Rx β ≠ Rx β * Rz α := by\n  intro α β h\n  intro hc\n  have := congrFun (congrFun hc 0) 2\n  rw [Rz_mul_Rx, Rx_mul_Rz] at this\n  simp at this\n  exact h (mul_eq_zero.2 this)\nexact @key (Real.pi / 6) (Real.pi / 2) ((by simp))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
