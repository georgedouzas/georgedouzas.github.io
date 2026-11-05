# name: ra-laplace-sin-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.laplace_sin file "ra-laplace-sin-test.sage"
# description: Sage's ∫₀^∞ e^(-ax) sin x dx = 1/(1 + a²)
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
instances = [O([str(av), '(by norm_num)'], integrate(exp(-av * x) * sin(x), x, 0, oo), 1 / (1 + SR(av) ** 2), edge=bool(av == 1)) for av in [1, 2, QQ(1) / 3]]
