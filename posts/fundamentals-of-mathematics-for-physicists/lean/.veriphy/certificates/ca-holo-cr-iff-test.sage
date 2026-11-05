# name: ca-holo-cr-iff-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyRiemann.cr_iff
# description: the Cauchy-Riemann criterion for w^2 (at 0 and 1 + i) and conj w; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w ^ 2)', '0', '(by fun_prop)'], 'have hf := (by fun_prop : DifferentiableAt ℝ (fun w : ℂ => w ^ 2) 0)\nrw [differentiableAt_complex_iff_differentiableAt_real, smul_eq_mul, Complex.ext_iff]\nsimp only [hf, true_and, mul_re, I_re, zero_mul, I_im, one_mul, zero_sub, mul_im, zero_add]\nconstructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩', True),
    (['(fun w : ℂ => w ^ 2)', '(1 + Complex.I)', '(by fun_prop)'], 'have hf := (by fun_prop : DifferentiableAt ℝ (fun w : ℂ => w ^ 2) (1 + Complex.I))\nrw [differentiableAt_complex_iff_differentiableAt_real, smul_eq_mul, Complex.ext_iff]\nsimp only [hf, true_and, mul_re, I_re, zero_mul, I_im, one_mul, zero_sub, mul_im, zero_add]\nconstructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩', False),
    (['(fun w : ℂ => (starRingEnd ℂ) w)', '1', '(Complex.differentiable_conj 1)'], 'have hf := (Complex.differentiable_conj 1)\nrw [differentiableAt_complex_iff_differentiableAt_real, smul_eq_mul, Complex.ext_iff]\nsimp only [hf, true_and, mul_re, I_re, zero_mul, I_im, one_mul, zero_sub, mul_im, zero_add]\nconstructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
