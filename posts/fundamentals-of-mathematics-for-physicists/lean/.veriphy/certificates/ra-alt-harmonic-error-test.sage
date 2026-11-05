# name: ra-alt-harmonic-error-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.alt_harmonic_error file "ra-alt-harmonic-error-test.sage"
# description: (-1)^m (log 2 - s_m) ≥ 0 and |s_m - log 2| < 1/(m+1) by 128-bit intervals at m = 0, 10, 11, 100, 101
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
for m in [0, 10, 11, 100, 101]:
    s = sum(QQ((-1) ** j) / (j + 1) for j in range(m))
    d = R(log(2)) - R(s)
    instances.append(O([str(m)], B(R((-1) ** m) * d >= 0 and abs(d) < R(1) / (m + 1)), QQ(1), edge=bool(m == 0)))
