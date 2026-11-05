# name: de-integrating-factor-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearFirstOrder.integrating_factor file "de-integrating-factor-test.sage"
# description: (μy)' = μ(y' + py) for μ = e^{∫p}: (p, y) = (2, sin), (x, x²), (0, 1) (edge), at x = 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('fun x => Real.exp (2 * x)', 'Real.sin', 'fun _ => 2', 'Real.cos 1', '1', '_', '_',
                    lhs=diff(exp(2*x)*sin(x), x).subs(x=1),
                    rhs=exp(2)*(cos(1) + 2*sin(1)), relation='==', edge=False),
    oracle_instance('fun x => Real.exp (x ^ 2 / 2)', 'fun x => x ^ 2', 'id', '2', '1', '_', '_',
                    lhs=diff(exp(x^2/2)*x^2, x).subs(x=1),
                    rhs=exp(1/2)*(2 + 1*1), relation='==', edge=False),
    oracle_instance('fun _ => 1', 'fun _ => 1', 'fun _ => 0', '0', '1', '_', '_',
                    lhs=diff(SR(1), x),
                    rhs=1*(0 + 0*1), relation='==', edge=True),
]
