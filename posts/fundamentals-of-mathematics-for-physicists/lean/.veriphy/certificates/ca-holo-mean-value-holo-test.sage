# name: ca-holo-mean-value-holo-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.holomorphic_mean_value
# description: f(z0) is the circle average for f = w, exp, w^2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '0', '1', '(by fun_prop : Differentiable ℂ (fun w : ℂ => w)).diffContOnCl'], 'exact (by fun_prop : Differentiable ℂ (fun w : ℂ => w)).diffContOnCl.circleAverage', True),
    (['Complex.exp', '1', '2', 'Complex.differentiable_exp.diffContOnCl'], 'exact Complex.differentiable_exp.diffContOnCl.circleAverage', False),
    (['(fun w : ℂ => w ^ 2)', 'Complex.I', '(1 / 2)', '(by fun_prop : Differentiable ℂ (fun w : ℂ => w ^ 2)).diffContOnCl'], 'exact (by fun_prop : Differentiable ℂ (fun w : ℂ => w ^ 2)).diffContOnCl.circleAverage', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
