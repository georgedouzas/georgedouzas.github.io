# name: la-integral-additive-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearMaps.integral_additive file "la-integral-additive-test.sage"
# description: ∫₀¹ (f + g) = ∫₀¹ f + ∫₀¹ g at f = g = 0 (kernel-checked) and at polynomial and trigonometric f, g, integrated exactly by Sage.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
x = SR.var('x')
PAIRS = [('(fun x => x)', '(fun x => x ^ 2)', 'continuous_id', '(continuous_pow 2)', x, x ** 2),
         ('Real.cos', '(fun x => 3 * x - 1)', 'Real.continuous_cos', '(by fun_prop)', cos(x), 3 * x - 1),
         ('Real.exp', 'Real.sin', 'Real.continuous_exp', 'Real.continuous_sin', exp(x), sin(x))]  # EDIT: the instances
instances = [kernel_instance('(fun _ => 0)', '(fun _ => 0)', 'continuous_const', 'continuous_const', proof='simp',
                             edge=True)]
for f, g, hf, hg, fe, ge in PAIRS:
    lhs = integrate(fe + ge, x, 0, 1)
    rhs = integrate(fe, x, 0, 1) + integrate(ge, x, 0, 1)
    instances.append(oracle_instance(f, g, hf, hg, lhs=lhs, rhs=rhs))
