# name: ra-integral-test-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.integral_test file "ra-integral-test-test.sage"
# description: the sum-integral comparison for decreasing f (1/x², 1/x, e^-x), by exact integrals and 128-bit intervals
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
instances = []
for f, n, edge in [(1 / x ** 2, 5, False), (1 / x, 10, False), (exp(-x), 3, False), (1 / x, 0, True)]:
    I_ = integrate(f, x, 1, 1 + n) if n else SR(0)
    lo = sum(f.subs(x=1 + k + 1) for k in range(n)); hi = sum(f.subs(x=1 + k) for k in range(n))
    instances.append(O([str(f), str(n), '‹_›'], B(RI(lo) <= RI(I_) and RI(I_) <= RI(hi)), QQ(1), edge=edge))
