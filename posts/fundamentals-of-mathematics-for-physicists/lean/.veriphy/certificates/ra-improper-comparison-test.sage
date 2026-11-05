# name: ra-improper-comparison-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.improper_comparison file "ra-improper-comparison-test.sage"
# description: the comparison test: Sage integrates the dominating g (1/x², 1/x³ + 1, 1/√x) and the dominated f
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
for f, g, a, b, edge in [(abs(sin(x)) / x ** 2, 1 / x ** 2, 1, oo, False), (1 / (1 + x ** 3), 1 / x ** 3, 1, oo, False),
                         (1 / (sqrt(x) * (1 + x)), 1 / sqrt(x), 0, 1, False), (SR(0), SR(0), 0, 1, True)]:
    Ig = integrate(g, x, a, b)
    If = integrate(f, x, a, b) if f != abs(sin(x)) / x ** 2 else None
    ok = RI(Ig) < 10 ** 6 and (If is None or RI(If) <= RI(Ig))
    instances.append(O([str(f), str(g), f'Set.Ioo {a} {b}', '‹_›', '‹_›', '‹_›', '‹_›', '‹_›'], B(ok), QQ(1), edge=edge))
