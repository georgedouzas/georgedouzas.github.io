# name: de-n-dim-harmonic-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.FundamentalSolutions.n_dim_harmonic file "de-n-dim-harmonic-test.sage"
# description: d/dr (r^{n-1} · A r^{1-n}) by Sage, exactly, at (n, A, r) = (3, 1, 1) (edge), (2, -2, 3/2), (5/2, 1/3, 2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('3', '1', '1', '_',
                    lhs=(diff(rs^((3)-1)*((1)*rs^(1-(3))), rs)).subs(rs=1).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '-2', '3/2', '_',
                    lhs=(diff(rs^((2)-1)*((-2)*rs^(1-(2))), rs)).subs(rs=3/2).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('5/2', '1/3', '2', '_',
                    lhs=(diff(rs^((5/2)-1)*((1/3)*rs^(1-(5/2))), rs)).subs(rs=2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
