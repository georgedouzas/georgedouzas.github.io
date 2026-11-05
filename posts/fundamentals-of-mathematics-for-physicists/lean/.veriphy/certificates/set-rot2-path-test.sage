# name: set-rot2-path-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Connected.rot2_path
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (α t : ℝ), (rot2 (t * α)).det = 1 ∧ rot2 (0 * α) = 1 ∧ rot2 (1 * α) = rot2 α := by\n  intro α t\n  refine ⟨?_, ?_, by rw [one_mul]⟩\n  · rw [rot2, Matrix.det_fin_two_of]; nlinarith [Real.sin_sq_add_cos_sq (t * α)]\n  · ext i j; fin_cases i <;> fin_cases j <;> simp [rot2]\nexact @key (0) (0)'),
    (['Real.pi', '1/2'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (α t : ℝ), (rot2 (t * α)).det = 1 ∧ rot2 (0 * α) = 1 ∧ rot2 (1 * α) = rot2 α := by\n  intro α t\n  refine ⟨?_, ?_, by rw [one_mul]⟩\n  · rw [rot2, Matrix.det_fin_two_of]; nlinarith [Real.sin_sq_add_cos_sq (t * α)]\n  · ext i j; fin_cases i <;> fin_cases j <;> simp [rot2]\nexact @key (Real.pi) (1/2)'),
    (['1', '2'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (α t : ℝ), (rot2 (t * α)).det = 1 ∧ rot2 (0 * α) = 1 ∧ rot2 (1 * α) = rot2 α := by\n  intro α t\n  refine ⟨?_, ?_, by rw [one_mul]⟩\n  · rw [rot2, Matrix.det_fin_two_of]; nlinarith [Real.sin_sq_add_cos_sq (t * α)]\n  · ext i j; fin_cases i <;> fin_cases j <;> simp [rot2]\nexact @key (1) (2)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
