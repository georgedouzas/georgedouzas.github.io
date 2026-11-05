# name: ca-holo-cube-uxx-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_uxx
# description: u_xx = 6x for u = x^3 - 3xy^2 at (x, y) = (0, 0), (1, -2), (-1/2, 3); each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '0'], 'rw [show (fun x => deriv (fun x => x ^ 3 - 3 * x * (0 : ℝ) ^ 2) x) = fun x => 3 * x ^ 2 - 3 * (0 : ℝ) ^ 2 from\n  funext fun x => KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_ux x (0)]\nrefine (((hasDerivAt_pow 2 (0 : ℝ)).const_mul 3).sub_const (3 * (0 : ℝ) ^ 2)).deriv.trans ?_\nnorm_num\ntry ring', True),
    (['1', '(-2)'], 'rw [show (fun x => deriv (fun x => x ^ 3 - 3 * x * (-2 : ℝ) ^ 2) x) = fun x => 3 * x ^ 2 - 3 * (-2 : ℝ) ^ 2 from\n  funext fun x => KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_ux x (-2)]\nrefine (((hasDerivAt_pow 2 (1 : ℝ)).const_mul 3).sub_const (3 * (-2 : ℝ) ^ 2)).deriv.trans ?_\nnorm_num\ntry ring', False),
    (['(-1 / 2)', '3'], 'rw [show (fun x => deriv (fun x => x ^ 3 - 3 * x * (3 : ℝ) ^ 2) x) = fun x => 3 * x ^ 2 - 3 * (3 : ℝ) ^ 2 from\n  funext fun x => KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_ux x (3)]\nrefine (((hasDerivAt_pow 2 (-1 / 2 : ℝ)).const_mul 3).sub_const (3 * (3 : ℝ) ^ 2)).deriv.trans ?_\nnorm_num\ntry ring', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
