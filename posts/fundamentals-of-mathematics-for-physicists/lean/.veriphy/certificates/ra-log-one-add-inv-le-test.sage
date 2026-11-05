# name: ra-log-one-add-inv-le-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.log_one_add_inv_le file "ra-log-one-add-inv-le-test.sage"
# description: log(1 + 1/n) ≤ 1/n by 128-bit intervals
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
instances = [O([str(n), '(by norm_num)'], R(log(1 + QQ(1) / n)), R(1) / n, rel='<=', edge=bool(n == 1)) for n in [1, 2, 10, 1000]]
