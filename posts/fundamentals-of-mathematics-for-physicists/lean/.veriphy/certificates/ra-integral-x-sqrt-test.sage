# name: ra-integral-x-sqrt-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.integral_x_sqrt file "ra-integral-x-sqrt-test.sage"
# description: Sage's ∫₀ᵃ x √(a² - x²) dx = a³/3 at a = 0, 1, 2, 1/2 (the chapter's value 1/3 at a = 1)
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
instances = [O([str(av), '(by norm_num)'], integrate(x * sqrt(av ** 2 - x ** 2), x, 0, av), SR(av) ** 3 / 3, edge=bool(av == 0))
             for av in [0, 1, 2, QQ(1) / 2]]
