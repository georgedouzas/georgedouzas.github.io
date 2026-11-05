# name: de-polar-r-neg-n-cos-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LaplaceEquation.polar_r_neg_n_cos file "de-polar-r-neg-n-cos-test.sage"
# description: the polar Laplacian of r⁻ⁿ cos nθ by Sage's derivatives, exactly, at (n, r, θ) = (0, 1, 0) (edge), (2, 1/2, 1), (3/2, 2, -1)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('0', '1', '0', '_',
                    lhs=(diff(rs^(-(0))*cos((0)*(0)), rs, 2) + 1/rs*diff(rs^(-(0))*cos((0)*(0)), rs) + 1/rs^2*diff(rs^(-(0))*cos((0)*ph), ph, 2).subs(ph=0)).subs(rs=1).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '1/2', '1', '_',
                    lhs=(diff(rs^(-(2))*cos((2)*(1)), rs, 2) + 1/rs*diff(rs^(-(2))*cos((2)*(1)), rs) + 1/rs^2*diff(rs^(-(2))*cos((2)*ph), ph, 2).subs(ph=1)).subs(rs=1/2).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('3/2', '2', '-1', '_',
                    lhs=(diff(rs^(-(3/2))*cos((3/2)*(-1)), rs, 2) + 1/rs*diff(rs^(-(3/2))*cos((3/2)*(-1)), rs) + 1/rs^2*diff(rs^(-(3/2))*cos((3/2)*ph), ph, 2).subs(ph=-1)).subs(rs=2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
