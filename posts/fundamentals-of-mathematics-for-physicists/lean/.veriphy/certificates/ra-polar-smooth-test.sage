# name: ra-polar-smooth-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.polar_smooth file "ra-polar-smooth-test.sage"
# description: Sage's partial derivatives up to order 3 of the polar map exist (finite) at sample points
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
r, ph = var('r ph')
F = [r * cos(ph), r * sin(ph)]
instances = []
for p, edge in [((0, 0), True), ((1, 2), False), ((-2, QQ(1) / 3), False)]:
    ok = all(RI(diff(c, r, a, ph, b).subs(r=p[0], ph=p[1]) if a or b else c.subs(r=p[0], ph=p[1])).absolute_diameter() < 1
             for c in F for a in range(3) for b in range(3 - a))
    instances.append(O([str(p)], B(ok), QQ(1), edge=edge))
