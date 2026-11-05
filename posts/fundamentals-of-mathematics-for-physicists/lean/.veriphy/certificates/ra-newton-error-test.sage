# name: ra-newton-error-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.newton_error file "ra-newton-error-test.sage"
# description: 0 ≤ a_n - √2 ≤ (a_n² - 2)/2 exactly (in ℚ(√2)), with a_6 - √2 < 10⁻⁴⁸
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
K2 = QuadraticField(2, 's')
s2 = K2.gen()
instances = []
a = QQ(2)
for n in range(7):
    d = K2(a) - s2
    ok = d >= 0 and d <= (a ** 2 - 2) / 2
    instances.append(O([str(n)], B(ok), QQ(1), edge=bool(n == 0)))
    if n == 6:
        assert RealIntervalField(400)(a) - RealIntervalField(400)(2).sqrt() < RealIntervalField(400)(10) ** -48
    a = (a + 2 / a) / 2
