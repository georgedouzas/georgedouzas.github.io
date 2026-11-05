# name: ra-log-series-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.log_series file "ra-log-series-test.sage"
# description: Sage sums ∑ (-1)^k x^(k+1)/(k+1) = log(1 + x), log 2 at x = 1
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
for x, edge in [(1, True), (QQ(1) / 2, False), (QQ(-1) / 2, False), (0, False)]:
    s = sum((-1) ** k * SR(x) ** (k + 1) / (k + 1), k, 0, oo) if x != 0 else SR(0)
    instances.append(O([str(x), '(by norm_num)', '(by norm_num)'], s, log(1 + x), edge=edge))
