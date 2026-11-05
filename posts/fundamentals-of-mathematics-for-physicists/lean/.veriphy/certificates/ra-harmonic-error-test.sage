# name: ra-harmonic-error-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.harmonic_error file "ra-harmonic-error-test.sage"
# description: 0 < H_n - log n - γ < 1/n by 128-bit intervals; at n = 10⁴ the difference is 5.0 · 10⁻⁵
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
for k in [1, 10, 100, 10 ** 4]:
    H = sum(QQ(1) / j for j in range(1, k + 1))
    d = R(H) - R(log(k)) - R(euler_gamma)
    instances.append(O([str(k), '(by norm_num)'], B(d > 0 and d < R(1) / k), QQ(1), edge=bool(k == 1)))
