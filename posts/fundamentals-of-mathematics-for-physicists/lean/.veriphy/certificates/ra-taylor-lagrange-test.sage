# name: ra-taylor-lagrange-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.taylor_lagrange file "ra-taylor-lagrange-test.sage"
# description: Sage solves for the Lagrange point c of e^x, sin x at sample x₀, x, n and checks c ∈ (x₀, x)
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
for f, x0, xv, n, edge in [(exp(x), 0, 1, 2, False), (sin(x), 0, 2, 3, False), (x ** 3, 0, 1, 2, True)]:
    T = sum(diff(f, x, k).subs(x=x0) / factorial(k) * (xv - x0) ** k for k in range(n + 1))
    rem = f.subs(x=xv) - T
    g = diff(f, x, n + 1) * (xv - x0) ** (n + 1) / factorial(n + 1)
    c = find_root(g - rem, x0 + 1e-9, xv - 1e-9) if bool(g.subs(x=x0) != g.subs(x=xv)) else (x0 + xv) / 2
    ok = RI(x0) < RI(c) < RI(xv) and abs(RI(g.subs(x=c)) - RI(rem)) < RI(10) ** -8
    instances.append(O([str(f), str(x0), str(xv), str(n), '(by norm_num)', '(by fun_prop)'], B(ok), QQ(1), edge=edge))
