# name: ca-holo-not-conformal-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.not_conformalAt_of_deriv_eq_zero
# description: no conformality at a zero of f': z^2 and z^3 at 0, a constant at i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w ^ 2)', '0', '(by fun_prop)', '(by simp)'], "rintro ⟨f', hd, hc⟩\nhave hf : DifferentiableAt ℂ (fun w : ℂ => w ^ 2) (0 : ℂ) := by fun_prop\nhave h : deriv (fun w : ℂ => w ^ 2) (0 : ℂ) = 0 := by simp\nhave h0 : HasFDerivAt (fun w : ℂ => w ^ 2) (0 : ℂ →L[ℝ] ℂ) (0 : ℂ) := by\n  have := hf.hasDerivAt.hasFDerivAt.restrictScalars ℝ\n  convert this using 1\n  ext w\n  simp [h]\nexact hc.ne_zero (hd.unique h0)", True),
    (['(fun w : ℂ => w ^ 3)', '0', '(by fun_prop)', '(by simp)'], "rintro ⟨f', hd, hc⟩\nhave hf : DifferentiableAt ℂ (fun w : ℂ => w ^ 3) (0 : ℂ) := by fun_prop\nhave h : deriv (fun w : ℂ => w ^ 3) (0 : ℂ) = 0 := by simp\nhave h0 : HasFDerivAt (fun w : ℂ => w ^ 3) (0 : ℂ →L[ℝ] ℂ) (0 : ℂ) := by\n  have := hf.hasDerivAt.hasFDerivAt.restrictScalars ℝ\n  convert this using 1\n  ext w\n  simp [h]\nexact hc.ne_zero (hd.unique h0)", False),
    (['(fun _ : ℂ => (1 : ℂ))', 'Complex.I', '(differentiableAt_const _)', '(by simp)'], "rintro ⟨f', hd, hc⟩\nhave hf : DifferentiableAt ℂ (fun _ : ℂ => (1 : ℂ)) (Complex.I) := differentiableAt_const _\nhave h : deriv (fun _ : ℂ => (1 : ℂ)) (Complex.I) = 0 := by simp\nhave h0 : HasFDerivAt (fun _ : ℂ => (1 : ℂ)) (0 : ℂ →L[ℝ] ℂ) (Complex.I) := by\n  have := hf.hasDerivAt.hasFDerivAt.restrictScalars ℝ\n  convert this using 1\n  ext w\n  simp [h]\nexact hc.ne_zero (hd.unique h0)", False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
