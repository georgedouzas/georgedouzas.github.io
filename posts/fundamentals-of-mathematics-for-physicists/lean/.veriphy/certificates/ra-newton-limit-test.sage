# name: ra-newton-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.newton_limit file "ra-newton-limit-test.sage"
# description: the positive fixed point of a ↦ (a + c/a)/2 is √c, and a_8 agrees with √c to 128 bits
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
instances = []
for c, edge in [(1, True), (2, False), (3, False), (10, False)]:
    a = QQ(c)
    for _ in range(8):
        a = (a + c / a) / 2
    instances.append(O([str(c), '(by norm_num)'], R(a), R(sqrt(c)), edge=edge))
