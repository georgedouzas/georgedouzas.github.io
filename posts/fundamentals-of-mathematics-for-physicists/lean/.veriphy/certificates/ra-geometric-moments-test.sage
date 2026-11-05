# name: ra-geometric-moments-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.geometric_moments file "ra-geometric-moments-test.sage"
# description: Sage sums ∑ k x^k and ∑ k(k-1) x^k; at x = 1/2: 2 and 4
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
for x in [0, QQ(1) / 2, QQ(-1) / 3]:
    a = sum(k * SR(x) ** k, k, 0, oo); b = sum(k * (k - 1) * SR(x) ** k, k, 0, oo)
    instances.append(O([str(x), '(by norm_num)'], B(bool(a == x / (1 - x) ** 2) and bool(b == 2 * x ** 2 / (1 - x) ** 3)), QQ(1), edge=bool(x == 0)))
