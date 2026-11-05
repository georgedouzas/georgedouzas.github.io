# name: ra-simplex-step-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.simplex_step file "ra-simplex-step-test.sage"
# description: Sage's ∫₀¹ (1 - x)^k/k! dx = 1/(k+1)!
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
instances = [O([str(kk)], integrate((1 - x) ** kk / factorial(kk), x, 0, 1), 1 / factorial(kk + 1), edge=bool(kk == 0)) for kk in [0, 1, 2, 5]]
