# name: set-cos-half-changes-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Quotients.cos_half_changes
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '(by simp)'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ) (h : Real.cos (φ / 2) ≠ 0), Real.cos ((φ + 2 * Real.pi) / 2) ≠ Real.cos (φ / 2) := by\n  intro φ h\n  rw [cos_half_antiperiodic φ]\n  intro h'\n  exact h (by linarith)\nexact @key (0) ((by simp))"),
    (['2 * Real.pi / 3', '(by rw [show 2 * Real.pi / 3 / 2 = Real.pi / 3 by ring, Real.cos_pi_div_three]; norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ) (h : Real.cos (φ / 2) ≠ 0), Real.cos ((φ + 2 * Real.pi) / 2) ≠ Real.cos (φ / 2) := by\n  intro φ h\n  rw [cos_half_antiperiodic φ]\n  intro h'\n  exact h (by linarith)\nexact @key (2 * Real.pi / 3) ((by rw [show 2 * Real.pi / 3 / 2 = Real.pi / 3 by ring, Real.cos_pi_div_three]; norm_num))"),
    (['-(2 * Real.pi / 3)', '(by rw [show -(2 * Real.pi / 3) / 2 = -(Real.pi / 3) by ring, Real.cos_neg, Real.cos_pi_div_three]; norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ) (h : Real.cos (φ / 2) ≠ 0), Real.cos ((φ + 2 * Real.pi) / 2) ≠ Real.cos (φ / 2) := by\n  intro φ h\n  rw [cos_half_antiperiodic φ]\n  intro h'\n  exact h (by linarith)\nexact @key (-(2 * Real.pi / 3)) ((by rw [show -(2 * Real.pi / 3) / 2 = -(Real.pi / 3) by ring, Real.cos_neg, Real.cos_pi_div_three]; norm_num))"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
