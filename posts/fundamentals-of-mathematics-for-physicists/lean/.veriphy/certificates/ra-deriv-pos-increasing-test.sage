# name: ra-deriv-pos-increasing-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.deriv_pos_increasing file "ra-deriv-pos-increasing-test.sage"
# description: Sage checks f' > 0 and f(x) < f(y) at sample points (eˣ, x³ + x, x + sin(x)/2)
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
for f, edge in [(exp(x), False), (x ** 3 + x, False), (x + sin(x) / 2, False), (x, True)]:
    pts = [-2, -1, 0, QQ(1) / 2, 3]
    ok = all(RI(diff(f, x).subs(x=p)) > 0 for p in pts) and all(RI(f.subs(x=pts[i])) < RI(f.subs(x=pts[i + 1])) for i in range(4))
    instances.append(O([str(f), '‹_›'], B(ok), QQ(1), edge=edge))
