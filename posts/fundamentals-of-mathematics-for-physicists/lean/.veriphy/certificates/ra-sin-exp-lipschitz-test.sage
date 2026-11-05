# name: ra-sin-exp-lipschitz-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.sin_exp_lipschitz file "ra-sin-exp-lipschitz-test.sage"
# description: |sin x - sin y| ≤ |x - y| and |eˣ - eʸ| ≤ e^max |x - y| by 128-bit intervals
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
for a, b, edge in [(0, 0, True), (1, 2, False), (-3, QQ(1) / 2, False), (5, -5, False)]:
    ok = abs(R(sin(a)) - R(sin(b))) <= abs(a - b) and abs(R(exp(a)) - R(exp(b))) <= R(exp(max(a, b))) * abs(a - b)
    instances.append(O([str(a), str(b)], B(ok), QQ(1), edge=edge))
