# name: ca-holo-conformal-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.conformal_of_deriv_ne_zero
# description: w, exp, w^2 are conformal where f' ≠ 0; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '0', '(by fun_prop)', '(by simp)'], 'exact (by fun_prop : DifferentiableAt ℂ ((fun w : ℂ => w)) (0 : ℂ)).conformalAt (by simp)', True),
    (['Complex.exp', '1', '(by fun_prop)', '(by simp)'], 'exact (by fun_prop : DifferentiableAt ℂ (Complex.exp) (1 : ℂ)).conformalAt (by simp)', False),
    (['(fun w : ℂ => w ^ 2)', '1', '(by fun_prop)', '(by simp)'], 'exact (by fun_prop : DifferentiableAt ℂ (fun w : ℂ => w ^ 2) (1 : ℂ)).conformalAt (by simp)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
