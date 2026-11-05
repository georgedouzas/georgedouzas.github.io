# name: ra-fundamental-theorem-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.fundamental_theorem file "ra-fundamental-theorem-test.sage"
# description: Sage's ∫ₐᵇ f = G(b) - G(a) and d/dx ∫ₐˣ f = f for sample f, G
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
x, s = var('x s')
instances = []
for f, G, a, b, xv, edge in [(cos(x), sin(x), 0, pi, 1, False), (x ** 2, x ** 3 / 3, 0, 2, 3, False), (exp(x), exp(x), -1, 1, 0, False), (SR(0), SR(0), 0, 1, 0, True)]:
    ftc2 = bool((integrate(f, x, a, b) - (G.subs(x=b) - G.subs(x=a))).simplify_full() == 0)
    assume(x > a)
    ftc1 = bool((diff(integrate(f.subs(x=s), s, a, x), x) - f).simplify_full() == 0)
    forget()
    instances.append(O([str(f), str(G), str(a), str(b), str(xv), '(by fun_prop)', '‹_›'], B(ftc1 and ftc2), QQ(1), edge=edge))
