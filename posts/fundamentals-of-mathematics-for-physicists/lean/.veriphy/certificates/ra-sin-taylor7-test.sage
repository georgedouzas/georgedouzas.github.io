# name: ra-sin-taylor7-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.sin_taylor7 file "ra-sin-taylor7-test.sage"
# description: |sin x - T₇(x)| ≤ |x|⁹/9! by 128-bit intervals at x = 0, 1, 2, 3 (3⁹/9! = 0.0542, error 0.0500 at 3)
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
for v in [0, 1, 2, 3, -5]:
    T7 = QQ(v) - QQ(v) ** 3 / 6 + QQ(v) ** 5 / 120 - QQ(v) ** 7 / 5040
    instances.append(O([str(v)], abs(R(sin(v)) - R(T7)), R(abs(QQ(v)) ** 9 / factorial(9)), rel='<=', edge=bool(v == 0)))
