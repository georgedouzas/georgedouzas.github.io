# name: ra-comparison-test-diverges-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.comparison_test_diverges file "ra-comparison-test-diverges-test.sage"
# description: divergence passes upward: 1/k ≤ 1/√k with both sums divergent (Sage)
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
def diverges(ex):
    try:
        s = sum(ex, k, 1, oo); return s in [oo] or 'Infinity' in str(s)
    except ValueError:
        return True
instances = [O(['(fun k : ℕ => 1 / ((k : ℝ) + 1))', '(fun k : ℕ => 1 / √((k : ℝ) + 1))', '‹_›', '‹_›', '‹_›'],
               B(diverges(1 / k) and diverges(1 / sqrt(k))), QQ(1)),
             O(['(fun k : ℕ => 1 / ((k : ℝ) + 1))', '(fun k : ℕ => 2 / ((k : ℝ) + 1))', '‹_›', '‹_›', '‹_›'],
               B(diverges(1 / k) and diverges(2 / k)), QQ(1)),
             O(['(fun k : ℕ => 1 / ((k : ℝ) + 1))', '(fun k : ℕ => 1 / ((k : ℝ) + 1))', '‹_›', '‹_›', '‹_›'],
               B(diverges(1 / k)), QQ(1), edge=True)]
