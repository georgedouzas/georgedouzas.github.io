# name: ra-poly-roots-le-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.poly_roots_le file "ra-poly-roots-le-test.sage"
# description: Sage counts the distinct real roots of sample polynomials: at most the degree
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
for P, Pl, edge in [(T ** 3 - T, 'Polynomial.X ^ 3 - Polynomial.X', False), (T ** 2 + 1, 'Polynomial.X ^ 2 + 1', False),
                    (T - 5, 'Polynomial.X - 5', True), ((T - 1) ** 2 * (T + 2), '(Polynomial.X - 1) ^ 2 * (Polynomial.X + 2)', False)]:
    instances.append(O([f'({Pl} : Polynomial ℝ)', '(by norm_num)'], QQ(len(P.roots(AA))), QQ(P.degree()), rel='<=', edge=edge))
