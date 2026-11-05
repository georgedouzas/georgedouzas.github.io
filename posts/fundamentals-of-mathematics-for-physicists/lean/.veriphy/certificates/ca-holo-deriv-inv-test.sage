# name: ca-holo-deriv-inv-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_inv
# description: (1/z)' = -(1/z)^2 at 1, i, -2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', 'one_ne_zero'], 'have hz := (one_ne_zero : (1 : ℂ) ≠ 0)\nshow deriv (fun w : ℂ => 1 / w) (1 : ℂ) = -(1 / (1 : ℂ)) ^ 2\nsimp only [one_div]\nrw [_root_.deriv_inv]\nring', True),
    (['Complex.I', 'Complex.I_ne_zero'], 'have hz := (Complex.I_ne_zero)\nshow deriv (fun w : ℂ => 1 / w) (Complex.I) = -(1 / (Complex.I)) ^ 2\nsimp only [one_div]\nrw [_root_.deriv_inv]\nring', False),
    (['(-2)', '(by norm_num)'], 'have hz := (by norm_num : (-2 : ℂ) ≠ 0)\nshow deriv (fun w : ℂ => 1 / w) (-2 : ℂ) = -(1 / (-2 : ℂ)) ^ 2\nsimp only [one_div]\nrw [_root_.deriv_inv]\nring', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
