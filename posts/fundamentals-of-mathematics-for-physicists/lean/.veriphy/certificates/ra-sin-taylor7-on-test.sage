# name: ra-sin-taylor7-on-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.sin_taylor7_on file "ra-sin-taylor7-on-test.sage"
# description: the remainder of T₇ is below 0.083 on [-π, π] (π⁹/9! = 0.0821) and 3 · 10⁻⁶ on [-1, 1], 128-bit intervals
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
for v, edge in [(pi, True), (1, False), (-2, False), (QQ(1) / 2, False)]:
    T7 = R(v) - R(v) ** 3 / 6 + R(v) ** 5 / 120 - R(v) ** 7 / 5040
    err = abs(R(sin(v)) - T7)
    ok = err < R(0.083) and (abs(R(v)) > 1 or err < R(3) / 10 ** 6)
    instances.append(O([str(v), '(by norm_num)' if v != pi else 'le_of_eq (abs_of_pos Real.pi_pos)'], B(ok), QQ(1), edge=edge))
