# name: ra-second-derivative-test-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.second_derivative_test file "ra-second-derivative-test-test.sage"
# description: Sage's f'(a) = 0, f''(a) > 0 and f(a ± h) > f(a) at small h (x², cos at π, the double well at 1)
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
for f, a, edge in [(x ** 2, 0, True), (cos(x), pi, False), ((x ** 2 - 1) ** 2 / 4, 1, False)]:
    ok = bool(diff(f, x).subs(x=a) == 0) and bool(diff(f, x, 2).subs(x=a) > 0) and         all(RI(f.subs(x=a + h)) > RI(f.subs(x=a)) for h in [QQ(1) / 100, -QQ(1) / 100, QQ(1) / 10 ** 4])
    instances.append(O([str(f), str(a), '(by fun_prop)', '‹_›', '‹_›'], B(ok), QQ(1), edge=edge))
