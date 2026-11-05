# name: ra-exp-taylor4-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.exp_taylor4 file "ra-exp-taylor4-test.sage"
# description: T₄(1) = 65/24, and 0 ≤ e^x - T₄(x) ≤ e^x x⁵/120 by 128-bit intervals (at x = 1: 0.0099 ≤ 0.0227)
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
for v in [0, 1, QQ(1) / 2, 2]:
    T4 = sum(QQ(v) ** k / factorial(k) for k in range(5))
    r = R(exp(v)) - R(T4)
    ok = r >= 0 and r <= R(exp(v)) * QQ(v) ** 5 / 120 and sum(QQ(1) / factorial(k) for k in range(5)) == QQ(65) / 24
    instances.append(O([str(v), '(by norm_num)'], B(ok), QQ(1), edge=bool(v == 0)))
