# name: ra-half-disc-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.half_disc file "ra-half-disc-test.sage"
# description: Sage's ∫₋ᵣʳ √(R² - x²) dx = πR²/2
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
instances = [O([str(Rr), '(by norm_num)'], integrate(sqrt(Rr ** 2 - x ** 2), x, -Rr, Rr), pi * SR(Rr) ** 2 / 2, edge=bool(Rr == 1)) for Rr in [1, 2, QQ(1) / 2]]
