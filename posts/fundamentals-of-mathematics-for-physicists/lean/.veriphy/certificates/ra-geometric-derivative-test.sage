# name: ra-geometric-derivative-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.geometric_derivative file "ra-geometric-derivative-test.sage"
# description: Sage sums ∑ (k+1) x^k = 1/(1 - x)²
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
instances = [O([str(x), '(by norm_num)'], 1 + sum((k + 1) * SR(x) ** k, k, 1, oo), 1 / (1 - x) ** 2, edge=bool(x == 0))
             for x in [0, QQ(1) / 2, QQ(-1) / 3, QQ(3) / 4]]
