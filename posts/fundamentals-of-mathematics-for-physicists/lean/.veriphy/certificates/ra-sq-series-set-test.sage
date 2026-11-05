# name: ra-sq-series-set-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.sq_series_set file "ra-sq-series-set-test.sage"
# description: ∑ x^(k+1)/(k+1)²: Sage sums it at x = ±1 (dilogarithm values), terms grow at x = 2
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
for x, edge in [(1, True), (-1, False), (QQ(1) / 2, False), (2, False)]:
    conv = abs(x) <= 1 and bool(abs(RI(polylog(2, x))) < 2)
    if abs(x) > 1:
        conv = not bool(limit(abs(x) ** (k + 1) / (k + 1) ** 2, k=oo) == 0)
        conv = False if conv else True
    instances.append(O([str(x)], B(conv), B(abs(x) <= 1), edge=edge))
