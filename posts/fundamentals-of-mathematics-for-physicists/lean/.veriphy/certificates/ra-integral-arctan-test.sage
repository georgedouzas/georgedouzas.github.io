# name: ra-integral-arctan-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.integral_arctan file "ra-integral-arctan-test.sage"
# description: Sage's ∫₋ₐᵃ dx/(1 + x²) = 2 arctan a; π/2 at a = 1
SEED = 0  # the instances are listed, no sampling
R = RealIntervalField(128)
def B(b):
    return QQ(1 if bool(b) else 0)
RI = RealIntervalField(256)
def O(values, lhs, rhs, rel='==', edge=False):
    # symbolic values (π, e, logs, limits Sage computed) are compared by 256-bit intervals; rationals exactly
    if any(hasattr(v, 'parent') and v.parent() is SR for v in (lhs, rhs)):
        lhs, rhs = RI(lhs), RI(rhs)
    return oracle_instance(*values, lhs=lhs, rhs=rhs, relation=rel, edge=edge)
x = var('x')
instances = [O([str(a)], integrate(1 / (1 + x ** 2), x, -a, a), 2 * arctan(SR(a)), edge=bool(a == 0)) for a in [0, 1, 2, -3]]
