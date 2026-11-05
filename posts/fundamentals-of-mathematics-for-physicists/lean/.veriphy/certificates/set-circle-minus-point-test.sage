# name: set-circle-minus-point-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Homeomorphism.circle_minus_point_connected
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ), IsConnected (circle \\ {wrap a}) := by\n  intro a\n  rw [← (wrap_bijOn a).image_eq]\n  exact (isConnected_Ioo (by linarith [Real.pi_pos])).image _\n    (show Continuous wrap by unfold wrap; fun_prop).continuousOn\nexact @key (0)'),
    (['Real.pi'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ), IsConnected (circle \\ {wrap a}) := by\n  intro a\n  rw [← (wrap_bijOn a).image_eq]\n  exact (isConnected_Ioo (by linarith [Real.pi_pos])).image _\n    (show Continuous wrap by unfold wrap; fun_prop).continuousOn\nexact @key (Real.pi)'),
    (['1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ), IsConnected (circle \\ {wrap a}) := by\n  intro a\n  rw [← (wrap_bijOn a).image_eq]\n  exact (isConnected_Ioo (by linarith [Real.pi_pos])).image _\n    (show Continuous wrap by unfold wrap; fun_prop).continuousOn\nexact @key (1)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
