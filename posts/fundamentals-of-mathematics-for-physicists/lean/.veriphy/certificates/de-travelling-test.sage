# name: de-travelling-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.WaveEquation.travelling_derivs file "de-travelling-test.sage"
# description: ∂ₜ²F(x + at) = a²F'' and ∂ₓ²F(x + at) = F'' by Sage for F = sin, exp(-s²), s³ (edge a = 0)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
t_ = var('t_'); s_ = var('s_')
instances = [
    oracle_instance('_', '_', '_', '_', '_', '2', '1', '1/2',
                    lhs=diff(sin(1/2 + 2*t_), t_, 2).subs(t_=1),
                    rhs=4*(-sin(1/2 + 2)), relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_', '-1', '1/3', '0',
                    lhs=diff(exp(-(x - t_)^2), x, 2).subs(x=0, t_=1/3),
                    rhs=diff(exp(-s_^2), s_, 2).subs(s_=-1/3), relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_', '0', '5', '2',
                    lhs=diff(SR(2)^3, t_, 2),
                    rhs=0^2*6*2, relation='==', edge=True),
]
