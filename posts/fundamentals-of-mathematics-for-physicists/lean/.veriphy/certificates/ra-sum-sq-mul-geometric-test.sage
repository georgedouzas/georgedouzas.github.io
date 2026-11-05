# name: ra-sum-sq-mul-geometric-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.sum_sq_mul_geometric file "ra-sum-sq-mul-geometric-test.sage"
# description: Sage sums ∑ n² x^n = x(1 + x)/(1 - x)³; 6 at x = 1/2
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
n = var('n')
instances = [O([str(x), '(by norm_num)'], sum(n ** 2 * SR(x) ** n, n, 0, oo), SR(x) * (1 + x) / (1 - x) ** 3, edge=bool(x == 0))
             for x in [0, QQ(1) / 2, QQ(-1) / 3, QQ(9) / 10]]
