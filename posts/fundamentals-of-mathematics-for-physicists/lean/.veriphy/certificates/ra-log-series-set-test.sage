# name: ra-log-series-set-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.log_series_set file "ra-log-series-set-test.sage"
# description: ∑ x^(k+1)/(k+1): Sage sums it at x = -1 (log 2 up to sign), finds the harmonic divergence at 1 and growing terms at 2
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
for x, edge in [(-1, True), (1, False), (QQ(1) / 2, False), (2, False)]:
    try:
        s = sum(SR(x) ** (k + 1) / (k + 1), k, 0, oo); conv = 'Infinity' not in str(s) and s.is_real()
    except ValueError:
        conv = False
    instances.append(O([str(x)], B(conv), B(-1 <= x < 1), edge=edge))
