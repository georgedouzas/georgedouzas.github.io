# name: set-ivt-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Connected.connected_image_ivt
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['id', 'continuous_id.continuousOn', '1/2', '(by simp; norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (f : ℝ → ℝ) (hf : ContinuousOn f (Icc 0 1)) (y : ℝ) (hy : y ∈ Icc (f 0) (f 1)), IsConnected (f '' Icc 0 1) ∧ (∃ x ∈ Icc (0 : ℝ) 1, f x = y) ∧ (f '' Icc 0 1).OrdConnected := by\n  intro f hf y hy\n  have hc : IsConnected (f '' Icc 0 1) := (isConnected_Icc zero_le_one).image f hf\n  exact ⟨hc, intermediate_value_Icc zero_le_one hf hy, hc.isPreconnected.ordConnected⟩\nexact @key (id) (continuous_id.continuousOn) (1/2) ((by simp; norm_num))"),
    (['fun _ => 0', 'continuousOn_const', '0', '(by simp)'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (f : ℝ → ℝ) (hf : ContinuousOn f (Icc 0 1)) (y : ℝ) (hy : y ∈ Icc (f 0) (f 1)), IsConnected (f '' Icc 0 1) ∧ (∃ x ∈ Icc (0 : ℝ) 1, f x = y) ∧ (f '' Icc 0 1).OrdConnected := by\n  intro f hf y hy\n  have hc : IsConnected (f '' Icc 0 1) := (isConnected_Icc zero_le_one).image f hf\n  exact ⟨hc, intermediate_value_Icc zero_le_one hf hy, hc.isPreconnected.ordConnected⟩\nexact @key (fun _ => 0) (continuousOn_const) (0) ((by simp))"),
    (['fun x => x ^ 2', '(continuous_pow 2).continuousOn', '1/4', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (f : ℝ → ℝ) (hf : ContinuousOn f (Icc 0 1)) (y : ℝ) (hy : y ∈ Icc (f 0) (f 1)), IsConnected (f '' Icc 0 1) ∧ (∃ x ∈ Icc (0 : ℝ) 1, f x = y) ∧ (f '' Icc 0 1).OrdConnected := by\n  intro f hf y hy\n  have hc : IsConnected (f '' Icc 0 1) := (isConnected_Icc zero_le_one).image f hf\n  exact ⟨hc, intermediate_value_Icc zero_le_one hf hy, hc.isPreconnected.ordConnected⟩\nexact @key (fun x => x ^ 2) ((continuous_pow 2).continuousOn) (1/4) ((by norm_num))"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
