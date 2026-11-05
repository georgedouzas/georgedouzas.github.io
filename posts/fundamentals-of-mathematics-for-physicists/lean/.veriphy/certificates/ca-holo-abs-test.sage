# name: ca-holo-abs-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.abs_not_differentiableAt
# description: |z| is not complex differentiable at z = 1, i, 1 + i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', '(by norm_num [Complex.ext_iff])'], 'intro h\nhave h2 : DifferentiableAt ℂ (fun w : ℂ => w * (starRingEnd ℂ) w) (1) := by\n  have e : (fun w : ℂ => w * (starRingEnd ℂ) w) = fun w => (‖w‖ : ℂ) * (‖w‖ : ℂ) := by\n    funext w\n    rw [Complex.mul_conj, Complex.normSq_eq_norm_sq]\n    push_cast\n    ring\n  rw [e]\n  exact h.mul h\nexact (by norm_num [Complex.ext_iff] : (1 : ℂ) ≠ 0) (((KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff (1)).1).1 h2)', True),
    (['Complex.I', '(by norm_num [Complex.ext_iff])'], 'intro h\nhave h2 : DifferentiableAt ℂ (fun w : ℂ => w * (starRingEnd ℂ) w) (Complex.I) := by\n  have e : (fun w : ℂ => w * (starRingEnd ℂ) w) = fun w => (‖w‖ : ℂ) * (‖w‖ : ℂ) := by\n    funext w\n    rw [Complex.mul_conj, Complex.normSq_eq_norm_sq]\n    push_cast\n    ring\n  rw [e]\n  exact h.mul h\nexact (by norm_num [Complex.ext_iff] : (Complex.I : ℂ) ≠ 0) (((KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff (Complex.I)).1).1 h2)', False),
    (['(1 + Complex.I)', '(by norm_num [Complex.ext_iff])'], 'intro h\nhave h2 : DifferentiableAt ℂ (fun w : ℂ => w * (starRingEnd ℂ) w) (1 + Complex.I) := by\n  have e : (fun w : ℂ => w * (starRingEnd ℂ) w) = fun w => (‖w‖ : ℂ) * (‖w‖ : ℂ) := by\n    funext w\n    rw [Complex.mul_conj, Complex.normSq_eq_norm_sq]\n    push_cast\n    ring\n  rw [e]\n  exact h.mul h\nexact (by norm_num [Complex.ext_iff] : (1 + Complex.I : ℂ) ≠ 0) (((KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff (1 + Complex.I)).1).1 h2)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
