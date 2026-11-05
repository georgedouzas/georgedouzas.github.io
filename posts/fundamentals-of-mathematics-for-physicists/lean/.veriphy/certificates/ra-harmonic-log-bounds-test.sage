# name: ra-harmonic-log-bounds-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.harmonic_log_bounds file "ra-harmonic-log-bounds-test.sage"
# description: log(n + 1) ≤ H_n ≤ 1 + log n by 128-bit intervals
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
for n in [1, 2, 10, 1000]:
    H = sum(QQ(1) / j for j in range(1, n + 1))
    instances.append(O([str(n)], B(R(log(n + 1)) <= R(H) and R(H) <= 1 + R(log(n))), QQ(1), edge=bool(n == 1)))
