# name: ra-monotone-riemann-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.monotone_riemann file "ra-monotone-riemann-test.sage"
# description: L_n ≤ ∫ f ≤ U_n and U_n - L_n = (f(b) - f(a))(b - a)/n for increasing f (x², eˣ, √x), exact or 128-bit
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
for f, a, b, n, edge in [(x ** 2, 0, 1, 4, False), (exp(x), 0, 1, 10, False), (sqrt(x), 0, 4, 8, False), (x, 0, 1, 1, True)]:
    h = QQ(b - a) / n
    L = sum(f.subs(x=a + k * h) * h for k in range(n)); U = sum(f.subs(x=a + (k + 1) * h) * h for k in range(n))
    I_ = integrate(f, x, a, b)
    ok = RI(L) <= RI(I_) and RI(I_) <= RI(U) and bool((U - L - (f.subs(x=b) - f.subs(x=a)) * (b - a) / n).simplify_full() == 0)
    instances.append(O([str(f), str(a), str(b), '(by norm_num)', '‹_›', str(n), '(by norm_num)'], B(ok), QQ(1), edge=edge))
