# name: ra-inversion-smooth-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.inversion_smooth file "ra-inversion-smooth-test.sage"
# description: Sage's partial derivatives of all orders ≤ 3 of the inversion are finite at sample points ≠ 0
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
u, v = var('u v')
I = [u / (u ** 2 + v ** 2), v / (u ** 2 + v ** 2)]
instances = []
for p, edge in [((1, 0), True), ((1, 2), False), ((-3, QQ(1) / 2), False)]:
    ok = all(c.subs(u=p[0], v=p[1]) in QQ for comp in I for a in range(4) for b in range(4 - a) for c in [diff(comp, u, a, v, b) if a or b else comp])
    instances.append(O([str(p), '(by norm_num)'], B(ok), QQ(1), edge=edge))
