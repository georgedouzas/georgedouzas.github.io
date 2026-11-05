# name: ra-cubic-unique-root-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.cubic_unique_root file "ra-cubic-unique-root-test.sage"
# description: Sage counts the real roots of x³ + x - y: exactly one
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
T = PolynomialRing(QQ, 't').gen()
instances = [O([str(y)], QQ(len((T ** 3 + T - y).roots(AA))), QQ(1), edge=bool(y == 0)) for y in [1, 0, -2, 100]]
