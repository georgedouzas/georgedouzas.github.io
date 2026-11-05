# name: ca-holo-entire-re-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.entire_re_harmonic
# description: Re of the entire w, exp, w^2 is harmonic; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '(by fun_prop)', '0'], 'exact ((by fun_prop : Differentiable ℂ (fun w : ℂ => w)).analyticAt _).harmonicAt_re', True),
    (['Complex.exp', 'Complex.differentiable_exp', '1'], 'exact (Complex.differentiable_exp.analyticAt _).harmonicAt_re', False),
    (['(fun w : ℂ => w ^ 2)', '(by fun_prop)', 'Complex.I'], 'exact ((by fun_prop : Differentiable ℂ (fun w : ℂ => w ^ 2)).analyticAt _).harmonicAt_re', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
