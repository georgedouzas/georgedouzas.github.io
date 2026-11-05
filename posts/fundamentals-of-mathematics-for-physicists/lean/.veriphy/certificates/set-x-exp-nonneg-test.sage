# name: set-x-exp-nonneg-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Compact.x_exp_nonneg
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', 'le_rfl'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ) (hx : 0 ≤ x), 0 ≤ x * Real.exp (-x) ∧ 0 * Real.exp (-0) = (0 : ℝ) := by\n  intro x hx\n  exact (⟨mul_nonneg hx (Real.exp_pos _).le, by simp⟩)\nexact @key (0) (le_rfl)'),
    (['1', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ) (hx : 0 ≤ x), 0 ≤ x * Real.exp (-x) ∧ 0 * Real.exp (-0) = (0 : ℝ) := by\n  intro x hx\n  exact (⟨mul_nonneg hx (Real.exp_pos _).le, by simp⟩)\nexact @key (1) ((by norm_num))'),
    (['2', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ) (hx : 0 ≤ x), 0 ≤ x * Real.exp (-x) ∧ 0 * Real.exp (-0) = (0 : ℝ) := by\n  intro x hx\n  exact (⟨mul_nonneg hx (Real.exp_pos _).le, by simp⟩)\nexact @key (2) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
