# name: ra-integral-cos-div-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.integral_cos_div file "ra-integral-cos-div-test.sage"
# description: Sage's ∫₀ᵃ cos x/(1 + sin² x) dx = arctan(sin a); π/4 at a = π/2
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
instances = [O([str(a)], integrate(cos(x) / (1 + sin(x) ** 2), x, 0, a), arctan(sin(a)), edge=bool(a == 0)) for a in [0, pi / 2, 1, -2]]
