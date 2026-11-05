# name: ra-nested-intervals-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.nested_intervals file "ra-nested-intervals-test.sage"
# description: Sage computes the common limit of the endpoints; it is the one point in all intervals
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
for a, ae, b, be, edge in [('(fun n : ℕ => -1 / ((n : ℝ) + 1))', -1 / (n + 1), '(fun n : ℕ => 1 / ((n : ℝ) + 1))', 1 / (n + 1), False),
                           ('(fun n : ℕ => 1 - 1 / ((n : ℝ) + 1))', 1 - 1 / (n + 1), '(fun n : ℕ => 1 + 1 / ((n : ℝ) + 1))', 1 + 1 / (n + 1), False),
                           ('(fun _ : ℕ => (2 : ℝ))', SR(2), '(fun _ : ℕ => (2 : ℝ))', SR(2), True)]:
    La, Lb = limit(ae, n=oo), limit(be, n=oo)
    inside = all(ae.subs(n=j) <= La <= be.subs(n=j) for j in range(50))
    instances.append(O([a, b, '‹_›', '‹_›', '‹_›', '‹_›'], B(bool(La == Lb) and inside), QQ(1), edge=edge))
