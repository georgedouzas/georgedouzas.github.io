# name: ca-holo-cayley-holo-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_differentiableAt
# description: the Cayley map is holomorphic at 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '(by simp)'], 'show DifferentiableAt ℂ (fun w : ℂ => (w - Complex.I) / (w + Complex.I)) (0 : ℂ)\nexact DifferentiableAt.div (by fun_prop) (by fun_prop) (by simp)', True),
    (['1', '(by norm_num [Complex.ext_iff])'], 'show DifferentiableAt ℂ (fun w : ℂ => (w - Complex.I) / (w + Complex.I)) (1 : ℂ)\nexact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff])', False),
    (['Complex.I', '(by norm_num [Complex.ext_iff])'], 'show DifferentiableAt ℂ (fun w : ℂ => (w - Complex.I) / (w + Complex.I)) (Complex.I)\nexact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff])', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
