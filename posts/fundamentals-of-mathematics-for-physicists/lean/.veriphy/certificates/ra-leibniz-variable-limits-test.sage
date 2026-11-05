# name: ra-leibniz-variable-limits-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.leibniz_variable_limits file "ra-leibniz-variable-limits-test.sage"
# description: Sage differentiates ∫ₓ^(x²) e^(-t²) dt (an erf expression) at sample x
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
F = integrate(exp(-s ** 2), s, x, x ** 2)
instances = [O([str(v)], diff(F, x).subs(x=v), (2 * x * exp(-x ** 4) - exp(-x ** 2)).subs(x=v), edge=bool(v == 0)) for v in [0, 1, -1, QQ(1) / 2]]
