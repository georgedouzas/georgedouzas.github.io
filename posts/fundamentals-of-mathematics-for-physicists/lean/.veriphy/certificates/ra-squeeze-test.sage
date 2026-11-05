# name: ra-squeeze-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.squeeze file "ra-squeeze-test.sage"
# description: squeezed sequences: Sage computes the common limit of the outer ones and the limit of the middle one
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
for a, ae, b, be, c, ce, edge in [
        ('(fun n : ℕ => -1 / ((n : ℝ) + 1))', -1 / (n + 1), '(fun n : ℕ => Real.sin n / ((n : ℝ) + 1))', sin(n) / (n + 1),
         '(fun n : ℕ => 1 / ((n : ℝ) + 1))', 1 / (n + 1), False),
        ('(fun n : ℕ => Real.cos (1 / ((n : ℝ) + 1)))', cos(1 / (n + 1)), '(fun n : ℕ => ((n : ℝ) + 1) * Real.sin (1 / ((n : ℝ) + 1)))',
         (n + 1) * sin(1 / (n + 1)), '(fun _ : ℕ => (1 : ℝ))', SR(1), False),
        ('(fun _ : ℕ => (0 : ℝ))', SR(0), '(fun _ : ℕ => (0 : ℝ))', SR(0), '(fun _ : ℕ => (0 : ℝ))', SR(0), True)]:
    La, Lc, Lb = limit(ae, n=oo), limit(ce, n=oo), limit(be, n=oo)
    instances.append(O([a, b, c, str(La), '‹_›', '‹_›', '‹_›', '‹_›'], Lb, La, edge=edge))
