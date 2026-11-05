# name: ra-integral-inv-sqrt-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.integral_inv_sqrt file "ra-integral-inv-sqrt-test.sage"
# description: Sage's ∫₀ᵃ dx/√x = 2√a; 2 at a = 1
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
instances = [O([str(a), '(by norm_num)'], integrate(1 / sqrt(x), x, 0, a) if a else SR(0), 2 * sqrt(SR(a)), edge=bool(a == 0)) for a in [0, 1, 4, 2]]
