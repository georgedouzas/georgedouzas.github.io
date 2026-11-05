# name: ra-cone-volume-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.cone_volume file "ra-cone-volume-test.sage"
# description: Sage's ∫₀ʰ π(Rz/h)² dz = πR²h/3 (π/3 at R = h = 1)
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
z = var('z')
instances = [O([str(Rr), str(hh), '(by norm_num)'], integrate(pi * (Rr * z / hh) ** 2, z, 0, hh), pi * Rr ** 2 * hh / 3, edge=bool(Rr == 1 and hh == 1)) for Rr, hh in [(1, 1), (2, 3), (QQ(1) / 2, 5)]]
