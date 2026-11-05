# name: ca-holo-deriv-fderiv-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyRiemann.deriv_eq_fderiv_one
# description: f' = u_x + i v_x for w, w^2, exp; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '0', '(by fun_prop)'], 'have h := (by fun_prop : DifferentiableAt ℂ ((fun w : ℂ => w)) (0 : ℂ))\nrw [h.fderiv_restrictScalars ℝ]\nsimp', True),
    (['(fun w : ℂ => w ^ 2)', 'Complex.I', '(by fun_prop)'], 'have h := (by fun_prop : DifferentiableAt ℂ (fun w : ℂ => w ^ 2) (Complex.I))\nrw [h.fderiv_restrictScalars ℝ]\nsimp', False),
    (['Complex.exp', '1', '(by fun_prop)'], 'have h := (by fun_prop : DifferentiableAt ℂ (Complex.exp) (1 : ℂ))\nrw [h.fderiv_restrictScalars ℝ]\nsimp', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
