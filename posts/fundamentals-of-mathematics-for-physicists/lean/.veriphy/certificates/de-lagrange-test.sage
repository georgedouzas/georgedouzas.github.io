# name: de-lagrange-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.lagrange_identity file "de-lagrange-test.sage"
# description: d/dx[p(uv' - u'v)] = u(pv')' - v(pu')' by Sage for (p, u, v) = (1, sin, sin 2x), (x, x², eˣ), (1, 0, 0) (edge), at x = 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
p_ = SR(1); u_ = sin(x); v_ = sin(2*x)
instances = [
    oracle_instance('_', '_', '_', '_', '_', '_', '_', '1', '_', '_', '_', '_',
                    lhs=(diff(u_*(p_*diff(v_,x)) - v_*(p_*diff(u_,x)), x) - (u_*diff(p_*diff(v_,x),x) - v_*diff(p_*diff(u_,x),x))).subs(x=1),
                    rhs=0, relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_', '_', '_', '1', '_', '_', '_', '_',
                    lhs=(diff(x^2*(x*diff(exp(x),x)) - exp(x)*(x*diff(x^2,x)), x) - (x^2*diff(x*diff(exp(x),x),x) - exp(x)*diff(x*diff(x^2,x),x))).subs(x=1),
                    rhs=0, relation='==', edge=False),
    oracle_instance('fun _ => 1', 'fun _ => 0', 'fun _ => 0', 'fun _ => 0', 'fun _ => 0', 'fun _ => 0', 'fun _ => 0', '0', '_', '_', '_', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
