# name: ra-leibniz-error-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.leibniz_error file "ra-leibniz-error-test.sage"
# description: |∑_{j<k} (-1)^j/(2j+1) - π/4| ≤ 1/(2k+1) by 128-bit intervals (k = 500 gives 10⁻³)
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
for k in [0, 1, 10, 500]:
    s = sum(QQ((-1) ** j) / (2 * j + 1) for j in range(k))
    instances.append(O([str(k)], abs(R(s) - R(pi) / 4), R(1) / (2 * k + 1), rel='<=', edge=bool(k == 0)))
