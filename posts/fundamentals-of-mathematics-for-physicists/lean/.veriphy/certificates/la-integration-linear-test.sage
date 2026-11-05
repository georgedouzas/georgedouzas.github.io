# name: la-integration-linear-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearFunctionals.integration_linear file "la-integration-linear-test.sage"
# description: ∫₀^{2π} (f + c g) = ∫ f + c ∫ g at f = g = 0 (kernel-checked) and at trigonometric and polynomial f, g, integrated exactly by Sage.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
x = SR.var('x')
CASES = [('Real.cos', '(fun x => x ^ 2)', '3', 'Real.continuous_cos', '(continuous_pow 2)', cos(x), x ** 2, 3),
         ('(fun x => Real.sin (2 * x))', '(fun _ => 1)', '-1 / 2', '(by fun_prop)', 'continuous_const', sin(2 * x), SR(1),
          QQ(-1) / 2),
         ('(fun x => Real.cos x ^ 2)', 'Real.sin', '2', '(by fun_prop)', 'Real.continuous_sin', cos(x) ** 2, sin(x), 2)]
instances = [kernel_instance('(fun _ => 0)', '(fun _ => 0)', '0', 'continuous_const', 'continuous_const', proof='simp',
                             edge=True)]
for f, g, c, hf, hg, fe, ge, ce in CASES:
    lhs = integrate(fe + ce * ge, x, 0, 2 * pi)
    rhs = integrate(fe, x, 0, 2 * pi) + ce * integrate(ge, x, 0, 2 * pi)
    instances.append(oracle_instance(f, g, c, hf, hg, lhs=lhs, rhs=rhs))
