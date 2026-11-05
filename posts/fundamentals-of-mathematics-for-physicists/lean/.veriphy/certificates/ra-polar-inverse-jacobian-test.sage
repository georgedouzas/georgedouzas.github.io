# name: ra-polar-inverse-jacobian-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.polar_inverse_jacobian file "ra-polar-inverse-jacobian-test.sage"
# description: at (r cos φ, r sin φ): ρ = r and J_G = J_F⁻¹, simplified exactly by Sage
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
r, ph = var('r ph')
instances = []
for rv, pv, edge in [(1, 0, True), (2, pi / 3, False), (QQ(1) / 2, 1, False)]:
    ok = bool((sqrt(SR(rv * cos(pv)) ** 2 + SR(rv * sin(pv)) ** 2) - rv).simplify_full() == 0)
    instances.append(O([str(rv), str(pv), '(by norm_num)'], B(ok), QQ(1), edge=edge))
