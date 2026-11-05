# name: ra-zeta-three-bounds-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.zeta_three_bounds file "ra-zeta-three-bounds-test.sage"
# description: S_N ≤ ζ(3) ≤ S_N + 1/(2N²) by 128-bit intervals; ζ(3) = 1.2020569…
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
for N in [1, 10, 100]:
    S = sum(QQ(1) / j ** 3 for j in range(1, N + 1))
    z = R(zeta(3))
    instances.append(O([str(N), '(by norm_num)'], B(R(S) <= z and z <= R(S) + R(1) / (2 * N ** 2)), QQ(1), edge=bool(N == 1)))
