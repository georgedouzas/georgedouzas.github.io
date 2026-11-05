# name: ra-binomial-series-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.binomial_series file "ra-binomial-series-test.sage"
# description: rigorous enclosures of ∑ C(α, k) x^k (80 terms plus a geometric tail bound) against (1 + x)^α
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
for a, x, edge in [(QQ(1) / 2, QQ(1) / 2, False), (-1, QQ(1) / 3, False), (QQ(-1) / 2, QQ(-1) / 4, False), (0, QQ(1) / 2, True)]:
    S = sum(binomial(a, j) * QQ(x) ** j for j in range(81))
    tail = 4 * abs(QQ(x)) ** 81 / (1 - abs(QQ(x))) * max(1, abs(QQ(a)) + 1) ** 2
    enc = RI(S - tail, S + tail)
    instances.append(O([str(a), str(x), '(by norm_num)'], B(enc.overlaps(RI((1 + QQ(x)) ** SR(a)))), QQ(1), edge=edge))
