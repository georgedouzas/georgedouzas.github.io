# name: ra-polar-jacobian-inverse-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.polar_jacobian_inverse file "ra-polar-jacobian-inverse-test.sage"
# description: Sage multiplies J⁻¹ J and simplifies to 1 (with cos² + sin² = 1)
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
for rv, pv, edge in [(1, 0, True), (2, pi / 3, False), (-3, 1, False)]:
    Ji = matrix(SR, [[cos(pv), sin(pv)], [-sin(pv) / rv, cos(pv) / rv]]); J = matrix(SR, [[cos(pv), -rv * sin(pv)], [sin(pv), rv * cos(pv)]])
    P = (Ji * J).simplify_full()
    instances.append(O([str(rv), str(pv), '(by norm_num)'], B(P == identity_matrix(2)), QQ(1), edge=edge))
