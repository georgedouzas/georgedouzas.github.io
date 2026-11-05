# name: set-exp-i-periodic-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Quotients.exp_I_periodic
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ), Complex.exp ((φ + 2 * Real.pi : ℝ) * Complex.I) = Complex.exp (φ * Complex.I) := by\n  intro φ\n  push_cast\n  rw [add_mul, Complex.exp_add, Complex.exp_two_pi_mul_I, mul_one]\nexact @key (0)'),
    (['1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ), Complex.exp ((φ + 2 * Real.pi : ℝ) * Complex.I) = Complex.exp (φ * Complex.I) := by\n  intro φ\n  push_cast\n  rw [add_mul, Complex.exp_add, Complex.exp_two_pi_mul_I, mul_one]\nexact @key (1)'),
    (['Real.pi'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (φ : ℝ), Complex.exp ((φ + 2 * Real.pi : ℝ) * Complex.I) = Complex.exp (φ * Complex.I) := by\n  intro φ\n  push_cast\n  rw [add_mul, Complex.exp_add, Complex.exp_two_pi_mul_I, mul_one]\nexact @key (Real.pi)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
