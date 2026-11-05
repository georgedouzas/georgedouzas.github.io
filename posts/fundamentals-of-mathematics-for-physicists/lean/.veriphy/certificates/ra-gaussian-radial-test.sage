# name: ra-gaussian-radial-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_radial file "ra-gaussian-radial-test.sage"
# description: Sage's ∫₀^∞ e^(-ar²) r dr = 1/(2a)
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
r = var('r')
instances = [O([str(av), '(by norm_num)'], integrate(exp(-av * r ** 2) * r, r, 0, oo), 1 / (2 * SR(av)), edge=bool(av == 1)) for av in [1, 2, QQ(1) / 2]]
