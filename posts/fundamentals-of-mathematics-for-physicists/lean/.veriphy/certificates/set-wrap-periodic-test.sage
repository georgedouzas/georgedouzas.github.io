# name: set-wrap-periodic-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Injective.wrap_periodic
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ), wrap (φ + 2 * Real.pi) = wrap φ := by\n  intro φ\n  simp [wrap]\nexact @key (0)'),
    (['1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ), wrap (φ + 2 * Real.pi) = wrap φ := by\n  intro φ\n  simp [wrap]\nexact @key (1)'),
    (['Real.pi / 3'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ), wrap (φ + 2 * Real.pi) = wrap φ := by\n  intro φ\n  simp [wrap]\nexact @key (Real.pi / 3)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
