# name: ra-p-series-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.p_series file "ra-p-series-test.sage"
# description: Sage decides convergence of ∑ 1/k^p (Sage's sum, or its 'divergent' error) and compares with p > 1
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
instances = []
for p, edge in [(1, True), (2, False), (QQ(1) / 2, False), (3, False)]:
    try:
        s = sum(1 / k ** p, k, 1, oo)
        conv = s.is_real() or bool(s < oo)
    except ValueError:
        conv = False
    instances.append(O([str(p)], B(conv), B(p > 1), edge=edge))
