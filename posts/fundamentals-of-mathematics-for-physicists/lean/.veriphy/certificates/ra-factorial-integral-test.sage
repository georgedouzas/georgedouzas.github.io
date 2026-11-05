# name: ra-factorial-integral-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.factorial_integral file "ra-factorial-integral-test.sage"
# description: Sage's ∫₀^∞ xⁿ e^(-ax) dx = n!/a^(n+1)
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
instances = [O([str(n), str(a), '(by norm_num)'], integrate(x ** n * exp(-a * x), x, 0, oo), factorial(n) / SR(a) ** (n + 1), edge=bool(n == 0))
             for n, a in [(0, 1), (1, 1), (5, 1), (3, 2), (2, QQ(1) / 2)]]
