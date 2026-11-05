# name: set-mem-circle-sphere-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.mem_circle_sphere
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '0', '0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y Z : ℝ), ((X, Y) ∈ circle ↔ X ^ 2 + Y ^ 2 = 1) ∧ (![X, Y, Z] ∈ sphere ↔ X ^ 2 + Y ^ 2 + Z ^ 2 = 1) := by\n  intro X Y Z\n  simp [circle, sphere]\nexact @key (1) (0) (0)'),
    (['3/5', '4/5', '0'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y Z : ℝ), ((X, Y) ∈ circle ↔ X ^ 2 + Y ^ 2 = 1) ∧ (![X, Y, Z] ∈ sphere ↔ X ^ 2 + Y ^ 2 + Z ^ 2 = 1) := by\n  intro X Y Z\n  simp [circle, sphere]\nexact @key (3/5) (4/5) (0)'),
    (['1', '1', '1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y Z : ℝ), ((X, Y) ∈ circle ↔ X ^ 2 + Y ^ 2 = 1) ∧ (![X, Y, Z] ∈ sphere ↔ X ^ 2 + Y ^ 2 + Z ^ 2 = 1) := by\n  intro X Y Z\n  simp [circle, sphere]\nexact @key (1) (1) (1)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
