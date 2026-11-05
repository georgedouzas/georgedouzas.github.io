# name: ra-third-order-f-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.third_order_f file "ra-third-order-f-test.sage"
# description: |x²y + sin y - (y + x²y - y³/6)| ≤ |y|⁴ · 5/96 for |y| ≤ 1, by 128-bit intervals
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
instances = []
for xv, yv, edge in [(0, 0, True), (1, 1, False), (2, QQ(-1) / 2, False), (5, QQ(1) / 10, False)]:
    lhs = abs(R(xv) ** 2 * yv + R(sin(yv)) - (R(yv) + R(xv) ** 2 * yv - R(yv) ** 3 / 6))
    instances.append(O([str(xv), str(yv), '(by norm_num)'], lhs, R(abs(QQ(yv)) ** 4 * QQ(5) / 96), rel='<=', edge=edge))
