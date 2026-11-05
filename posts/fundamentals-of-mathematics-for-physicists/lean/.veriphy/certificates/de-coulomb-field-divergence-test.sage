# name: de-coulomb-field-divergence-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.DeltaFunction.coulomb_field_divergence file "de-coulomb-field-divergence-test.sage"
# description: ∇·(x/r³) by Sage's partial derivatives, summed exactly, at (x, y, z) = (1, 0, 0) (edge), (1, 2, 2), (-1/2, 1, 3/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('1', '0', '0', '_',
                    lhs=(diff(xs/(xs^2+ys^2+zs^2)^(3/2), xs) + diff(ys/(xs^2+ys^2+zs^2)^(3/2), ys) + diff(zs/(xs^2+ys^2+zs^2)^(3/2), zs)).subs(xs=1, ys=0, zs=0).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1', '2', '2', '_',
                    lhs=(diff(xs/(xs^2+ys^2+zs^2)^(3/2), xs) + diff(ys/(xs^2+ys^2+zs^2)^(3/2), ys) + diff(zs/(xs^2+ys^2+zs^2)^(3/2), zs)).subs(xs=1, ys=2, zs=2).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('-1/2', '1', '3/2', '_',
                    lhs=(diff(xs/(xs^2+ys^2+zs^2)^(3/2), xs) + diff(ys/(xs^2+ys^2+zs^2)^(3/2), ys) + diff(zs/(xs^2+ys^2+zs^2)^(3/2), zs)).subs(xs=-1/2, ys=1, zs=3/2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
