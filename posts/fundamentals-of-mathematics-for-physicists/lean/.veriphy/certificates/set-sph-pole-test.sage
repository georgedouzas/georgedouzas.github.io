# name: set-sph-pole-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Injective.sph_pole
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ), sph 0 φ = northPole := by\n  intro φ\n  ext i; fin_cases i <;> simp [sph, northPole]\nexact @key (0)'),
    (['1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ), sph 0 φ = northPole := by\n  intro φ\n  ext i; fin_cases i <;> simp [sph, northPole]\nexact @key (1)'),
    (['Real.pi'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ), sph 0 φ = northPole := by\n  intro φ\n  ext i; fin_cases i <;> simp [sph, northPole]\nexact @key (Real.pi)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
