# name: ra-p-integral-infty-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.p_integral_infty file "ra-p-integral-infty-test.sage"
# description: Sage decides ∫₁^∞ x^-p (convergent: 1/(p-1); 'divergent' error otherwise)
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
for p, edge in [(1, True), (2, False), (3, False), (QQ(1) / 2, False)]:
    try:
        v = integrate(x ** (-p), x, 1, oo); conv = True
    except ValueError:
        conv = False
    instances.append(O([str(p)], B(conv), B(p > 1), edge=edge))
