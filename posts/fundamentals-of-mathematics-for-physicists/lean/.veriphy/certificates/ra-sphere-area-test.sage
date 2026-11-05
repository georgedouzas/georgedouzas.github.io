# name: ra-sphere-area-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.sphere_area file "ra-sphere-area-test.sage"
# description: n π^(n/2)/Γ(n/2 + 1) = 2π^(n/2)/Γ(n/2) exactly by Sage (2π, 4π, 2π², …)
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
instances = [O([str(n), '(by norm_num)'], n * pi ** (QQ(n) / 2) / gamma(QQ(n) / 2 + 1), 2 * pi ** (QQ(n) / 2) / gamma(QQ(n) / 2), edge=bool(n == 1)) for n in [1, 2, 3, 4, 7]]
