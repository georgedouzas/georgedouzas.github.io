# name: de-dalembert-example-solves-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PDEComputations.dalembert_example_solves file "de-dalembert-example-solves-test.sage"
# description: u_tt - c² u_xx for d'Alembert's solution of the Gaussian-plus-cosine data by Sage's partial derivatives, in interval arithmetic, at (c, t, x) = (1, 0, 0) (edge), (2, 1/2, 1), (1/3, 1, -2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('1', '0', '0', '_',
                    lhs=RIF(abs((diff(((exp(-(xs-(1)*tt)^2) + exp(-(xs+(1)*tt)^2))/2 + cos(xs)*sin((1)*tt)/(1)), tt, 2) - (1)^2*diff(((exp(-(xs-(1)*tt)^2) + exp(-(xs+(1)*tt)^2))/2 + cos(xs)*sin((1)*tt)/(1)), xs, 2)).subs(tt=0, xs=0))),
                    rhs=RIF(10^-15), relation='<', edge=True),
    oracle_instance('2', '1/2', '1', '_',
                    lhs=RIF(abs((diff(((exp(-(xs-(2)*tt)^2) + exp(-(xs+(2)*tt)^2))/2 + cos(xs)*sin((2)*tt)/(2)), tt, 2) - (2)^2*diff(((exp(-(xs-(2)*tt)^2) + exp(-(xs+(2)*tt)^2))/2 + cos(xs)*sin((2)*tt)/(2)), xs, 2)).subs(tt=1/2, xs=1))),
                    rhs=RIF(10^-15), relation='<', edge=False),
    oracle_instance('1/3', '1', '-2', '_',
                    lhs=RIF(abs((diff(((exp(-(xs-(1/3)*tt)^2) + exp(-(xs+(1/3)*tt)^2))/2 + cos(xs)*sin((1/3)*tt)/(1/3)), tt, 2) - (1/3)^2*diff(((exp(-(xs-(1/3)*tt)^2) + exp(-(xs+(1/3)*tt)^2))/2 + cos(xs)*sin((1/3)*tt)/(1/3)), xs, 2)).subs(tt=1, xs=-2))),
                    rhs=RIF(10^-15), relation='<', edge=False),
]
