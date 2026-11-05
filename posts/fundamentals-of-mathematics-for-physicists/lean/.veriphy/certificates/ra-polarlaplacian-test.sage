# name: ra-polarlaplacian-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.PolarLaplacian file "ra-polarlaplacian-test.sage"
# description: Sage computes both sides of Δu = u_rr + u_r/r + u_φφ/r² for polynomial and exponential u
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
x, y, r, ph = var('x y r ph')
instances = []
for u, pt, edge in [(x ** 2 + y ** 2, (1, 0), False), (x ** 3 * y, (2, QQ(1) / 3), False), (exp(x) * sin(y), (1, 1), False), (SR(1), (1, 0), True)]:
    lap = (diff(u, x, 2) + diff(u, y, 2)).subs(x=r * cos(ph), y=r * sin(ph))
    v = u.subs(x=r * cos(ph), y=r * sin(ph))
    pol = diff(v, r, 2) + diff(v, r) / r + diff(v, ph, 2) / r ** 2
    ok = bool((lap - pol).subs(r=pt[0], ph=pt[1]).simplify_full() == 0) or abs(RI((lap - pol).subs(r=pt[0], ph=pt[1]))) < RI(10) ** -60
    instances.append(O([f'(fun x y => {u})', '(by fun_prop)', str(pt[0]), str(pt[1]), '(by norm_num)'], B(ok), QQ(1), edge=edge))
