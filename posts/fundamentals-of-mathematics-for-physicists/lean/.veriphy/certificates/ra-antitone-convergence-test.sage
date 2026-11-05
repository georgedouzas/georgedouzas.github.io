# name: ra-antitone-convergence-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.antitone_convergence file "ra-antitone-convergence-test.sage"
# description: Sage computes the limits of decreasing bounded sequences and checks they are below the terms (the infimum)
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
for a, ex, edge in [('(fun n : ℕ => 1 / ((n : ℝ) + 1))', 1 / (n + 1), False),
                    ('(fun n : ℕ => (n + 2 : ℝ) / (n + 1))', (n + 2) / (n + 1), False),
                    ('(fun _ : ℕ => (3 : ℝ))', SR(3), True)]:
    L = limit(ex, n=oo)
    inf_ok = all(R(L) <= R(ex.subs(n=j)) for j in range(0, 200))
    instances.append(O([a, '‹_›', '‹_›'], B(inf_ok), QQ(1), edge=edge))
