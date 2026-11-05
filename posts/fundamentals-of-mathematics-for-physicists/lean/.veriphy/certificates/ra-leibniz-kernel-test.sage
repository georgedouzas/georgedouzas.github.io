# name: ra-leibniz-kernel-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.leibniz_kernel file "ra-leibniz-kernel-test.sage"
# description: Sage differentiates ∫₀ˣ (x - t) f(t) dt and compares with ∫₀ˣ f, for f = cos, eᵗ, t²
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
for f, xv, edge in [(cos(s), 1, False), (exp(s), 2, False), (s ** 2, -1, False), (SR(0), 0, True)]:
    assume(x > 0)
    G = integrate((x - s) * f, s, 0, x)
    forget()
    lhs = diff(G, x).subs(x=xv); rhs = integrate(f, s, 0, xv)
    instances.append(O([f'(fun t => {f})', '(by fun_prop)', str(xv)], lhs, rhs, edge=edge))
