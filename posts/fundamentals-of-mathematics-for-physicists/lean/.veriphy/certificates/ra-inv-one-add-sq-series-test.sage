# name: ra-inv-one-add-sq-series-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.inv_one_add_sq_series file "ra-inv-one-add-sq-series-test.sage"
# description: Sage sums ∑ (-1)^k x^(2k) = 1/(1 + x²) for |x| < 1, and the terms do not tend to 0 for |x| ≥ 1
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
for x in [0, QQ(1) / 2, 1, 2]:
    if abs(x) < 1:
        ok = x == 0 or bool(sum((-1) ** k * SR(x) ** (2 * k), k, 0, oo) == 1 / (1 + x ** 2))
    else:
        ok = abs(x) ** 2 >= 1   # |(-1)^k x^(2k)| = |x|^(2k) ≥ 1
    instances.append(O([str(x)], B(ok), QQ(1), edge=bool(x == 1)))
