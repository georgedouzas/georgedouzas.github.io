# name: ra-cauchyhadamard-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.CauchyHadamard file "ra-cauchyhadamard-test.sage"
# description: Sage computes ρ = lim |c_k|^(1/k) and the root test at x inside and outside 1/ρ
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
k = var('k')
instances = []
for c, ce, x, edge in [('(fun k : ℕ => (2 : ℝ) ^ k)', 2 ** k, QQ(1) / 4, False), ('(fun k : ℕ => (2 : ℝ) ^ k)', 2 ** k, 1, False),
                       ('(fun k : ℕ => (3 : ℝ) ^ k * ((k : ℝ) + 1))', 3 ** k * (k + 1), QQ(1) / 6, False),
                       ('(fun _ : ℕ => (1 : ℝ))', SR(1), 2, True)]:
    rho = limit(abs(ce) ** (1 / k), k=oo)
    root = limit(abs(ce * x ** k) ** (1 / k), k=oo)
    ok = (abs(x) < 1 / rho and bool(root < 1)) or (abs(x) > 1 / rho and bool(root > 1))
    instances.append(O([c, str(rho), '(by norm_num)', '‹_›', str(x)], B(ok), QQ(1), edge=edge))
