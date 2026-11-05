# name: ra-nroot-bound-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.nroot_bound file "ra-nroot-bound-test.sage"
# description: 0 ≤ n^(1/n) - 1 and (n^(1/n) - 1)² ≤ 2/(n - 1), exactly in algebraic numbers
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
for k in [2, 3, 10, 100]:
    h = AA(k) ** (QQ(1) / k) - 1
    instances.append(O([str(k), '(by norm_num)'], B(h >= 0 and h ** 2 <= QQ(2) / (k - 1)), QQ(1), edge=bool(k == 2)))
