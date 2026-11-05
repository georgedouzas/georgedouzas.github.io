# name: ra-smooth-two-variables-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.smooth_two_variables file "ra-smooth-two-variables-test.sage"
# description: Sage's partials up to order 3 of log(x² + y²) (off 0) and (x² - y², 2xy) are finite at sample points
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
x, y = var('x y')
instances = []
for p, edge in [((1, 0), True), ((1, 2), False), ((-3, QQ(1) / 2), False)]:
    fs = [log(x ** 2 + y ** 2), x ** 2 - y ** 2, 2 * x * y]
    ok = all(RI(diff(c, x, a, y, b).subs(x=p[0], y=p[1]) if a or b else c.subs(x=p[0], y=p[1])).absolute_diameter() < 1
             for c in fs for a in range(3) for b in range(3 - a))
    instances.append(O([str(p)], B(ok), QQ(1), edge=edge))
