# name: ra-cos-derivatives-diverge-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.cos_derivatives_diverge file "ra-cos-derivatives-diverge-test.sage"
# description: cos(n(2k+1)π) = (-1)ⁿ: Sage's limit of the sequence does not exist
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
n = var('n')
instances = []
for kk in [0, 1, 5]:
    vals = [cos(j * (2 * kk + 1) * pi) for j in range(6)]
    instances.append(O([str(kk)], B(vals == [1, -1, 1, -1, 1, -1]), QQ(1), edge=bool(kk == 0)))
