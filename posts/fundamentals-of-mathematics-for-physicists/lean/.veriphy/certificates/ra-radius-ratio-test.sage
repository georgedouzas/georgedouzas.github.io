# name: ra-radius-ratio-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.radius_ratio file "ra-radius-ratio-test.sage"
# description: Sage computes R = lim |c_k/c_(k+1)| and decides convergence inside and divergence outside (1/k, 1/k², 2^k)
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
for c, ce, R0, x, edge in [('(fun k : ℕ => 1 / ((k : ℝ) + 1))', 1 / (k + 1), 1, QQ(1) / 2, False),
                           ('(fun k : ℕ => 1 / ((k : ℝ) + 1) ^ 2)', 1 / (k + 1) ** 2, 1, QQ(3) / 2, False),
                           ('(fun k : ℕ => (2 : ℝ) ^ k)', 2 ** k, QQ(1) / 2, QQ(1) / 4, False),
                           ('(fun _ : ℕ => (1 : ℝ))', SR(1), 1, 0, True)]:
    Rk = limit(abs(ce / ce.subs(k=k + 1)), k=oo)
    ratio = limit(abs(ce.subs(k=k + 1) * x ** (k + 1) / (ce * x ** k)), k=oo) if x != 0 else 0
    ok = bool(Rk == R0) and ((abs(x) < R0 and bool(ratio < 1)) or (abs(x) > R0 and bool(ratio > 1)))
    instances.append(O([c, str(R0), '(by norm_num)', '‹_›', '‹_›', str(x)], B(ok), QQ(1), edge=edge))
