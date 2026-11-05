# name: ra-gaussian-moment-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.gaussian_moment file "ra-gaussian-moment-test.sage"
# description: Sage's even moments ∫ x^(2n) e^(-ax²) dx against (2n-1)!!/(2a)ⁿ √(π/a)
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
instances = [O([str(n), str(a), '(by norm_num)'], integrate(x ** (2 * n) * exp(-a * x ** 2), x, -oo, oo),
               (prod(range(1, 2 * n, 2)) if n > 0 else 1) / (2 * SR(a)) ** n * sqrt(pi / a), edge=bool(n == 0))
             for n, a in [(0, 1), (1, 1), (2, 1), (3, 1), (1, 2)]]
