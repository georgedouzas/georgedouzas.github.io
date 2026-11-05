# name: ra-erf-deriv-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.erf_deriv file "ra-erf-deriv-test.sage"
# description: Sage's derivative of ∫₀ˣ e^(-t²) dt = √π/2 erf(x) is e^(-x²)
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
assume(x > 0)
F = integrate(exp(-s ** 2), s, 0, x)   # √π/2 erf(x)
forget()
instances = [O([str(v)], diff(F, x).subs(x=v), exp(-SR(v) ** 2), edge=bool(v == 0)) for v in [0, 1, -2, QQ(1) / 2]]
