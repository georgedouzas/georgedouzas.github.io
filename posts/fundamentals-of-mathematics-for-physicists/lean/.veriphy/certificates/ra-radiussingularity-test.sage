# name: ra-radiussingularity-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.RadiusSingularity file "ra-radiussingularity-test.sage"
# description: Sage computes the radius from the coefficients (root test on 40 coefficients) and the distance to the nearest complex root
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
x = var('x')
instances = []
for P, x0, xv, edge in [('1 + x^2', 0, QQ(1) / 2, True), ('x - 2', 0, 1, False), ('(x - 2)*(x^2 + 9)', 0, 1, False), ('(x - 2)*(x^2 + 9)', 1, QQ(3) / 2, False)]:
    p = SR(P.replace('^', '**'))
    dist = min(abs(CC(r) - x0) for r in p.roots(x, ring=QQbar, multiplicities=False))
    tt = var('tt')
    ser = (1 / p.subs(x=tt + x0)).taylor(tt, 0, 60)
    coeffs = [ser.coefficient(tt, n) for n in range(40, 61)]
    est = max(abs(RR(c)) ** (-1 / n) for c, n in zip(coeffs, range(40, 61)) if c != 0)
    instances.append(O([f'({P.replace("x", "Polynomial.X")} : Polynomial ℂ)', str(x0), '(by norm_num)', str(xv), '(by norm_num)'], B(abs(est - RR(dist)) < 0.15 * RR(dist)), QQ(1), edge=edge))
