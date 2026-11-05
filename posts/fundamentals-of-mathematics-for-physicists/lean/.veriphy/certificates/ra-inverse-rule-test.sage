# name: ra-inverse-rule-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.inverse_rule file "ra-inverse-rule-test.sage"
# description: Sage differentiates the inverse (log of exp, cube root of x³, arctan of tan) and compares with 1/f'
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
x, y = var('x y')
instances = []
for f, g, a, edge in [(exp(x), log(y), 1, False), (x ** 3, y ** (QQ(1) / 3), 2, False), (tan(x), arctan(y), QQ(1) / 3, False), (x, y, 5, True)]:
    lhs = diff(g, y).subs(y=f.subs(x=a)); rhs = 1 / diff(f, x).subs(x=a)
    instances.append(O([str(f), str(g), str(a), str(rhs), '‹_›', '‹_›', '‹_›', '‹_›', '‹_›'], lhs, rhs, edge=edge))
