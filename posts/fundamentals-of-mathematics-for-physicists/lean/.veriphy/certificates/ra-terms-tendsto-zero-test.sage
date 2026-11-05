# name: ra-terms-tendsto-zero-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.terms_tendsto_zero file "ra-terms-tendsto-zero-test.sage"
# description: for convergent series (2^-k, 1/k², (-1/3)^k) Sage computes the sum (finite) and lim a_k = 0
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
for a, ex, edge in [('(fun k : ℕ => (1 / 2 : ℝ) ^ k)', (QQ(1) / 2) ** k, False), ('(fun k : ℕ => 1 / ((k : ℝ) + 1) ^ 2)', 1 / (k + 1) ** 2, False),
                    ('(fun _ : ℕ => (0 : ℝ))', 0 * k, True)]:
    s = sum(ex, k, 0, oo)
    instances.append(O([a, '‹_›'], B(s in SR and limit(ex, k=oo) == 0), QQ(1), edge=edge))
