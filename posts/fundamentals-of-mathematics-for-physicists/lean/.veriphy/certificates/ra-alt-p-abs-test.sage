# name: ra-alt-p-abs-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.alt_p_abs file "ra-alt-p-abs-test.sage"
# description: Sage sums ∑ 1/k^p (ζ(p)) for p = 2, 3, 4: finite
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
k = var('k')
instances = [O([str(p), '(by norm_num)'], B(sum(1 / k ** p, k, 1, oo) == zeta(p)), QQ(1), edge=bool(p == 2)) for p in [2, 3, 4]]
