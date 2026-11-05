# name: de-euler-log-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparationOfVariables.euler_log file "de-euler-log-test.sage"
# description: r² (log r)'' + r (log r)' by Sage's derivatives, exactly, at r = 1 (edge), 2, 1/3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('1', '_',
                    lhs=(rs^2*diff(log(rs), rs, 2) + rs*diff(log(rs), rs)).subs(rs=1).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '_',
                    lhs=(rs^2*diff(log(rs), rs, 2) + rs*diff(log(rs), rs)).subs(rs=2).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('1/3', '_',
                    lhs=(rs^2*diff(log(rs), rs, 2) + rs*diff(log(rs), rs)).subs(rs=1/3).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
