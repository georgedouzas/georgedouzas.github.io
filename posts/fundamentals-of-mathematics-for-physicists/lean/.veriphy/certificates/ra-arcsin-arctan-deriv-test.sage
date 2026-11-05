# name: ra-arcsin-arctan-deriv-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.arcsin_arctan_deriv file "ra-arcsin-arctan-deriv-test.sage"
# description: Sage's derivatives of arcsin and arctan at sample points
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
for v in [0, QQ(1) / 2, QQ(-1) / 3]:
    ok = bool(diff(arcsin(x), x).subs(x=v) == 1 / sqrt(1 - v ** 2)) and bool(diff(arctan(x), x).subs(x=v) == 1 / (1 + v ** 2))
    instances.append(O([str(v), '(by norm_num)', '(by norm_num)'], B(ok), QQ(1), edge=bool(v == 0)))
