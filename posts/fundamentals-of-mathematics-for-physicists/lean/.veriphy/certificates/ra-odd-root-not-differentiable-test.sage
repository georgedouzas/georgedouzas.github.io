# name: ra-odd-root-not-differentiable-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.odd_root_not_differentiable file "ra-odd-root-not-differentiable-test.sage"
# description: Sage's limit of x^(1/(2k+1))/x at 0⁺ is +∞: the odd root has no derivative at 0
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
instances = [O([str(kk), '(by norm_num)', f'(the real {2 * kk + 1}-th root)', '‹_›'], B(limit(x ** (1 / QQ(2 * kk + 1)) / x, x=0, dir='+') == oo), QQ(1), edge=bool(kk == 1)) for kk in [1, 2, 5]]
