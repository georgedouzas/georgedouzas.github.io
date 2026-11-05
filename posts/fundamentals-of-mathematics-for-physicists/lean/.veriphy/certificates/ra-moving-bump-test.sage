# name: ra-moving-bump-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.moving_bump file "ra-moving-bump-test.sage"
# description: the bump n·1_(0,1/n): f_m(x) = 0 once 1/(m+1) ≤ x, and Sage's ∫₀¹ = n · 1/n = 1
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
for n, xv in [(1, QQ(1) / 2), (5, QQ(1) / 100), (10, 0), (3, QQ(-1) / 2)]:
    eventually_zero = xv <= 0 or any(QQ(1) / (m + 1) <= xv for m in range(1000))
    instances.append(O([str(n), '(by norm_num)', str(xv)], B(eventually_zero and n * (QQ(1) / n) == 1), QQ(1), edge=bool(xv == 0)))
