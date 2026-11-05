# name: ra-p-integral-finite-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.p_integral_finite file "ra-p-integral-finite-test.sage"
# description: Sage's ∫₁ᵇ x^-p = (b^(1-p) - 1)/(1 - p) and ∫₁ᵇ dx/x = log b
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
for p, b, edge in [(2, 3, False), (QQ(1) / 2, 4, False), (3, 1, True), (-1, 2, False)]:
    ok = bool((integrate(x ** (-p), x, 1, b) - (SR(b) ** (1 - p) - 1) / (1 - p)).simplify_full() == 0) and \
        bool((integrate(1 / x, x, 1, b) - log(SR(b))).simplify_full() == 0)
    instances.append(O([str(p), str(b), '(by norm_num)', '(by norm_num)'], B(ok), QQ(1), edge=edge))
