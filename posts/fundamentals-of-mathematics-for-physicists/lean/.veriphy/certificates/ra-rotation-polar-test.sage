# name: ra-rotation-polar-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.rotation_polar file "ra-rotation-polar-test.sage"
# description: Sage's R_α J_F(r, φ) - J_F(r, φ + α) simplifies to 0 at sample values
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
for rv, pv, av, edge in [(1, 0, 0, True), (2, pi / 3, 1, False), (QQ(1) / 2, 1, pi / 6, False)]:
    Ra = matrix(SR, [[cos(av), -sin(av)], [sin(av), cos(av)]]); J = lambda p: matrix(SR, [[cos(p), -rv * sin(p)], [sin(p), rv * cos(p)]])
    D = (Ra * J(pv) - J(pv + av)).simplify_full()
    ok = all(bool(D[i, j].simplify_trig().simplify_full() == 0) or abs(RI(D[i, j])) < RI(10) ** -60 for i in range(2) for j in range(2))
    instances.append(O([str(rv), str(pv), str(av)], B(ok), QQ(1), edge=edge))
