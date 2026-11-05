# name: ra-rearrangement-invariant-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.rearrangement_invariant file "ra-rearrangement-invariant-test.sage"
# description: Sage sums ∑ a_(e(n)) and ∑ a_n exactly for permutations e swapping finitely many indices
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
for a, ex, swap, edge in [('(fun k : ℕ => (-1 / 2 : ℝ) ^ k)', (QQ(-1) / 2) ** k, (0, 1), False),
                          ('(fun k : ℕ => (1 / 3 : ℝ) ^ k)', (QQ(1) / 3) ** k, (2, 5), False),
                          ('(fun _ : ℕ => (0 : ℝ))', 0 * k, (0, 0), True)]:
    total = sum(ex, k, 0, oo)
    i, j = swap
    permuted = total - ex.subs(k=i) - ex.subs(k=j) + ex.subs(k=j) + ex.subs(k=i)   # swapping two terms
    instances.append(O([a, '‹_›', f'(Equiv.swap {i} {j})'], permuted, total, edge=edge))
