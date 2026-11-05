# name: ca-holo-exp-ray-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_horizontal_ray
# description: arg e^{x + iy0} = y0 at (0, 0), (1, 1), (-1, 3); each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '0', '(by linarith [Real.pi_gt_three])', '(by linarith [Real.pi_gt_three])'], 'have e : Complex.exp (((0 : ℝ) : ℂ) + ((0 : ℝ) : ℂ) * Complex.I) = (Real.exp (0) : ℂ) * (Complex.cos ((0 : ℝ) : ℂ) + Complex.sin ((0 : ℝ) : ℂ) * Complex.I) := by\n  rw [Complex.exp_add, Complex.exp_mul_I, Complex.ofReal_exp]\nrw [e]\nexact Complex.arg_mul_cos_add_sin_mul_I (Real.exp_pos _) ⟨by linarith [Real.pi_gt_three], by linarith [Real.pi_gt_three]⟩', True),
    (['1', '1', '(by linarith [Real.pi_gt_three])', '(by linarith [Real.pi_gt_three])'], 'have e : Complex.exp (((1 : ℝ) : ℂ) + ((1 : ℝ) : ℂ) * Complex.I) = (Real.exp (1) : ℂ) * (Complex.cos ((1 : ℝ) : ℂ) + Complex.sin ((1 : ℝ) : ℂ) * Complex.I) := by\n  rw [Complex.exp_add, Complex.exp_mul_I, Complex.ofReal_exp]\nrw [e]\nexact Complex.arg_mul_cos_add_sin_mul_I (Real.exp_pos _) ⟨by linarith [Real.pi_gt_three], by linarith [Real.pi_gt_three]⟩', False),
    (['(-1)', '3', '(by linarith [Real.pi_gt_three])', '(by linarith [Real.pi_gt_three])'], 'have e : Complex.exp (((-1 : ℝ) : ℂ) + ((3 : ℝ) : ℂ) * Complex.I) = (Real.exp (-1) : ℂ) * (Complex.cos ((3 : ℝ) : ℂ) + Complex.sin ((3 : ℝ) : ℂ) * Complex.I) := by\n  rw [Complex.exp_add, Complex.exp_mul_I, Complex.ofReal_exp]\nrw [e]\nexact Complex.arg_mul_cos_add_sin_mul_I (Real.exp_pos _) ⟨by linarith [Real.pi_gt_three], by linarith [Real.pi_gt_three]⟩', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
