# name: ra-gaussian-linear-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.gaussian_linear file "ra-gaussian-linear-test.sage"
# description: Sage's ∫ e^(-ax² + bx) dx = √(π/a) e^(b²/4a)
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
instances = [O([str(a), str(b), '(by norm_num)'], integrate(exp(-a * x ** 2 + b * x), x, -oo, oo), sqrt(pi / a) * exp(SR(b) ** 2 / (4 * a)), edge=bool(b == 0))
             for a, b in [(1, 0), (1, 1), (2, -3), (QQ(1) / 2, 2)]]
