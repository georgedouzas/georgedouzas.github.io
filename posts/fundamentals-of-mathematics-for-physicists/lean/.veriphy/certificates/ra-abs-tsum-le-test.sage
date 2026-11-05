# name: ra-abs-tsum-le-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.abs_tsum_le file "ra-abs-tsum-le-test.sage"
# description: Sage sums ∑ a_k and ∑ |a_k| exactly and compares
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
for a, ex, edge in [('(fun k : ℕ => (-1 / 2 : ℝ) ^ k)', (QQ(-1) / 2) ** k, False), ('(fun k : ℕ => (-1 : ℝ) ^ k / ((k : ℝ) + 1) ^ 2)', (-1) ** k / (k + 1) ** 2, False),
                    ('(fun _ : ℕ => (0 : ℝ))', 0 * k, True)]:
    s = sum(ex, k, 0, oo); t_ = sum(abs(ex), k, 0, oo)
    instances.append(O([a, '‹_›'], abs(s), t_, rel='<=', edge=edge))
