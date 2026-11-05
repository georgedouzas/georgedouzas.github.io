# name: ra-feynmantrick-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.FeynmanTrick file "ra-feynmantrick-test.sage"
# description: Sage's exact ∫₀¹ (xᵃ - 1)/log x dx = log(1 + a) (log 2 at a = 1, log 4 at a = 3)
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
for av in [1, 3, 2]:
    v = integrate((x ** av - 1) / log(x), x, 0, 1)
    instances.append(O([str(av), '(by norm_num)'], v, log(1 + SR(av)), edge=bool(av == 1)))
