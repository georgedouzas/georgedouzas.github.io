# name: ra-integral-cauchy-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.integral_cauchy file "ra-integral-cauchy-test.sage"
# description: Sage's ∫_ℝ dx/(c² + x²) = π/c; π at c = 1
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
instances = [O([str(c), '(by norm_num)'], integrate(1 / (c ** 2 + x ** 2), x, -oo, oo), pi / c, edge=bool(c == 1)) for c in [1, 2, QQ(1) / 2]]
