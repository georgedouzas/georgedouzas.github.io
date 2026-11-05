# name: ra-sq-lt-bounds-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.sq_lt_bounds file "ra-sq-lt-bounds-test.sage"
# description: the real roots ±√p of t² - p (exact algebraic numbers) bound {t² < p} and are not in it, at p = 1, 2, 3
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
T = PolynomialRing(QQ, 't').gen()
instances = []
for p, edge in [(1, True), (2, False), (3, False)]:
    roots = sorted(r for r, m in (T ** 2 - p).roots(AA))
    assert roots[0] ** 2 == p and roots[1] ** 2 == p and roots[0] == -roots[1]
    instances.append(O([str(p), '(by norm_num)'], B(roots[1] == AA(p).sqrt()), QQ(1), edge=edge))
