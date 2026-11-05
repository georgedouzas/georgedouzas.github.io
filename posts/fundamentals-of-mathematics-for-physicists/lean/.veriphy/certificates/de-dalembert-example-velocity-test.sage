# name: de-dalembert-example-velocity-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PDEComputations.dalembert_example_velocity file "de-dalembert-example-velocity-test.sage"
# description: u_t at t = 0 for d'Alembert's solution of the Gaussian-plus-cosine data by Sage, exactly, at (c, x) = (1, 0) (edge), (2, 1), (-1/2, 3)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('1', '0', '_',
                    lhs=(diff(((exp(-(xs-(1)*tt)^2) + exp(-(xs+(1)*tt)^2))/2 + cos(xs)*sin((1)*tt)/(1)), tt).subs(tt=0) - cos(xs)).subs(xs=0).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '1', '_',
                    lhs=(diff(((exp(-(xs-(2)*tt)^2) + exp(-(xs+(2)*tt)^2))/2 + cos(xs)*sin((2)*tt)/(2)), tt).subs(tt=0) - cos(xs)).subs(xs=1).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('-1/2', '3', '_',
                    lhs=(diff(((exp(-(xs-(-1/2)*tt)^2) + exp(-(xs+(-1/2)*tt)^2))/2 + cos(xs)*sin((-1/2)*tt)/(-1/2)), tt).subs(tt=0) - cos(xs)).subs(xs=3).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
