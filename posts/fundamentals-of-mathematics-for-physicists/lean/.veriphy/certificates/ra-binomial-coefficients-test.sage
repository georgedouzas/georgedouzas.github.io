# name: ra-binomial-coefficients-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.binomial_coefficients file "ra-binomial-coefficients-test.sage"
# description: Sage's binomial(-1, k) = (-1)^k and binomial(1/2, 1) = 1/2, binomial(1/2, 2) = -1/8
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
instances = [O([str(k)], B(binomial(-1, k) == (-1) ** k and binomial(QQ(1) / 2, 1) == QQ(1) / 2 and binomial(QQ(1) / 2, 2) == QQ(-1) / 8), QQ(1),
               edge=bool(k == 0)) for k in [0, 1, 2, 7]]
