# name: ca-holo-exp-conformal-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_conformal
# description: exp is conformal with (e^z)' = e^z ≠ 0 at 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'exact ⟨by simp, Complex.exp_ne_zero _, Complex.differentiableAt_exp.conformalAt (by simpa using Complex.exp_ne_zero _)⟩', True),
    (['1'], 'exact ⟨by simp, Complex.exp_ne_zero _, Complex.differentiableAt_exp.conformalAt (by simpa using Complex.exp_ne_zero _)⟩', False),
    (['Complex.I'], 'exact ⟨by simp, Complex.exp_ne_zero _, Complex.differentiableAt_exp.conformalAt (by simpa using Complex.exp_ne_zero _)⟩', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
