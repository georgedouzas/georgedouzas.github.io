# name: ra-sin-div-bounds-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.sin_div_bounds file "ra-sin-div-bounds-test.sage"
# description: cos x ≤ sin x / x ≤ 1 on (0, 1] by 128-bit intervals
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
instances = [O([str(v), '(by norm_num)', '(by norm_num)'], B(R(cos(v)) <= R(sin(v)) / R(v) and R(sin(v)) / R(v) <= 1), QQ(1),
               edge=bool(v == 1)) for v in [1, QQ(1) / 2, QQ(1) / 10, QQ(1) / 1000]]
