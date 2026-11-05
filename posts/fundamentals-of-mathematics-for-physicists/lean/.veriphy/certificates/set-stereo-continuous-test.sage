# name: set-stereo-continuous-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Homeomorphism.stereo_continuousOn
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ), ContinuousOn (fun p : Fin 3 → ℝ => (p 0 / (c - p 2), p 1 / (c - p 2))) {p | p 2 ≠ c} := by\n  intro c\n  have h : ∀ p ∈ {p : Fin 3 → ℝ | p 2 ≠ c}, c - p 2 ≠ 0 := fun p hp => sub_ne_zero.2 (Ne.symm hp)\n  refine ContinuousOn.prodMk ?_ ?_\n  · exact ((continuous_apply 0).continuousOn).div (by fun_prop) h\n  · exact ((continuous_apply 1).continuousOn).div (by fun_prop) h\nexact @key (1)'),
    (['0'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ), ContinuousOn (fun p : Fin 3 → ℝ => (p 0 / (c - p 2), p 1 / (c - p 2))) {p | p 2 ≠ c} := by\n  intro c\n  have h : ∀ p ∈ {p : Fin 3 → ℝ | p 2 ≠ c}, c - p 2 ≠ 0 := fun p hp => sub_ne_zero.2 (Ne.symm hp)\n  refine ContinuousOn.prodMk ?_ ?_\n  · exact ((continuous_apply 0).continuousOn).div (by fun_prop) h\n  · exact ((continuous_apply 1).continuousOn).div (by fun_prop) h\nexact @key (0)'),
    (['-1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ), ContinuousOn (fun p : Fin 3 → ℝ => (p 0 / (c - p 2), p 1 / (c - p 2))) {p | p 2 ≠ c} := by\n  intro c\n  have h : ∀ p ∈ {p : Fin 3 → ℝ | p 2 ≠ c}, c - p 2 ≠ 0 := fun p hp => sub_ne_zero.2 (Ne.symm hp)\n  refine ContinuousOn.prodMk ?_ ?_\n  · exact ((continuous_apply 0).continuousOn).div (by fun_prop) h\n  · exact ((continuous_apply 1).continuousOn).div (by fun_prop) h\nexact @key (-1)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
