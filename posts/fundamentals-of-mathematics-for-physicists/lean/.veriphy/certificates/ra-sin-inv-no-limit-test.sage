# name: ra-sin-inv-no-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.sin_inv_no_limit file "ra-sin-inv-no-limit-test.sage"
# description: Sage's limit of sin(1/x) at 0 is 'ind' (no limit), for every candidate L
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
x = var('x')
lim = limit(sin(1 / x), x=0)
instances = [O([str(L)], B(str(lim) == 'ind'), QQ(1), edge=bool(L == 0)) for L in [0, 1, -1, QQ(1) / 2]]
