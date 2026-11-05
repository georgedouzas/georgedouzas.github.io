# name: ra-harmonic-sub-log-decreasing-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.harmonic_sub_log_decreasing file "ra-harmonic-sub-log-decreasing-test.sage"
# description: H_n - log n decreases and stays positive, by 128-bit intervals
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
for k in [1, 2, 10, 1000]:
    H = sum(QQ(1) / j for j in range(1, k + 1))
    a = R(H) - R(log(k))
    b = R(H + QQ(1) / (k + 1)) - R(log(k + 1))
    instances.append(O([str(k), '(by norm_num)'], B(b < a and a > 0), QQ(1), edge=bool(k == 1)))
