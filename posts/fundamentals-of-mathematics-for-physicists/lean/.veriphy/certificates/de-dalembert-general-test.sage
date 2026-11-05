# name: de-dalembert-general-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.WaveEquation.dalembert_general file "de-dalembert-general-test.sage"
# description: f(x - ct) + g(x + ct) solves the wave equation by Sage for generic f, g (symbolic functions), and for c = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
t_, c_ = var('t_ c_'); ff = function('ff'); gg = function('gg')
instances = [
    oracle_instance('_', '_', '_', '_', '_', '_', '_', '_', '_', '_', 'c_', 't_', 'x',
                    lhs=(diff(ff(x - c_*t_) + gg(x + c_*t_), t_, 2) - c_^2*diff(ff(x - c_*t_) + gg(x + c_*t_), x, 2)).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('Real.sin', '_', '_', 'Real.exp', '_', '_', '_', '_', '_', '_', '2', '1', '0',
                    lhs=(diff(sin(x - 2*t_) + exp(x + 2*t_), t_, 2) - 4*diff(sin(x - 2*t_) + exp(x + 2*t_), x, 2)).subs(x=0, t_=1),
                    rhs=0, relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '0', '1', '1',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
