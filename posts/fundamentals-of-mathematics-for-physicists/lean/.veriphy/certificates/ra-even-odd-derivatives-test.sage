# name: ra-even-odd-derivatives-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.even_odd_derivatives file "ra-even-odd-derivatives-test.sage"
# description: Sage's odd derivatives of even functions (cos, x² + 1, cosh) and even ones of odd functions vanish at 0
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
for f, k, edge in [(cos(x), 0, True), (cos(x), 2, False), (x ** 4 + x ** 2 + 1, 1, False), (sin(x), 1, False)]:
    ev = bool((f.subs(x=-x) - f).simplify_full() == 0); od = bool((f.subs(x=-x) + f).simplify_full() == 0)
    ok = (not ev or diff(f, x, 2 * k + 1).subs(x=0) == 0) and (not od or diff(f, x, 2 * k).subs(x=0) == 0)
    instances.append(O([str(f), str(k)], B(ok), QQ(1), edge=edge))
