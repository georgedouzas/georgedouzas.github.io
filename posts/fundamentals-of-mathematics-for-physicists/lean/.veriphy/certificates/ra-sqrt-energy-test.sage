# name: ra-sqrt-energy-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.sqrt_energy file "ra-sqrt-energy-test.sage"
# description: m + p²/2m - p⁴/8m³ ≤ √(m² + p²) ≤ m + p²/2m exactly in algebraic numbers
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
for m, p, edge in [(1, 0, True), (1, 1, False), (2, QQ(1) / 2, False), (QQ(1) / 3, 1, False)]:
    m, p = QQ(m), QQ(p)
    s = AA(m ** 2 + p ** 2).sqrt()
    instances.append(O([str(m), str(p), '(by norm_num)'], B(s <= m + p ** 2 / (2 * m) and m + p ** 2 / (2 * m) - p ** 4 / (8 * m ** 3) <= s), QQ(1), edge=edge))
