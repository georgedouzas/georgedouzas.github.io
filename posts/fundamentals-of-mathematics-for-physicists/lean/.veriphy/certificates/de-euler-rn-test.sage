# name: de-euler-rn-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparationOfVariables.euler_rn file "de-euler-rn-test.sage"
# description: r² (rⁿ)'' + r (rⁿ)' - n² rⁿ by Sage's derivatives, exactly, at (n, r) = (0, 1) (edge), (2, 3), (3/2, 1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('0', '1', '_',
                    lhs=(rs^2*diff(rs^(0), rs, 2) + rs*diff(rs^(0), rs) - (0)^2*rs^(0)).subs(rs=1).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '3', '_',
                    lhs=(rs^2*diff(rs^(2), rs, 2) + rs*diff(rs^(2), rs) - (2)^2*rs^(2)).subs(rs=3).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('3/2', '1/2', '_',
                    lhs=(rs^2*diff(rs^(3/2), rs, 2) + rs*diff(rs^(3/2), rs) - (3/2)^2*rs^(3/2)).subs(rs=1/2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
