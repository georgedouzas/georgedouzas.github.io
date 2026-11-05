# name: ra-limit-div-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.limit_div file "ra-limit-div-test.sage"
# description: the quotient rule: Sage computes lim a, lim b and lim (a / b) and compares
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
SEQ = {'(fun n : ℕ => 1 / ((n : ℝ) + 1))': 1 / (n + 1), '(fun n : ℕ => (2 * (n : ℝ) + 1) / (n + 3))': (2 * n + 1) / (n + 3),
       '(fun n : ℕ => (1 + 1 / ((n : ℝ) + 1)) ^ (n + 1))': (1 + 1 / (n + 1)) ** (n + 1),
       '(fun _ : ℕ => (2 : ℝ))': SR(2), '(fun n : ℕ => (-1 : ℝ) ^ n / ((n : ℝ) + 1))': (-1) ** n / (n + 1),
       '(fun n : ℕ => (3 * (n : ℝ) ^ 2 + 1) / ((n : ℝ) ^ 2 + 5))': (3 * n ** 2 + 1) / (n ** 2 + 5)}
def lim(t):
    return limit(SEQ[t], n=oo)

instances = []
pairs = [('(fun _ : ℕ => (2 : ℝ))', '(fun _ : ℕ => (2 : ℝ))', True),
         ('(fun n : ℕ => 1 / ((n : ℝ) + 1))', '(fun n : ℕ => (2 * (n : ℝ) + 1) / (n + 3))', False),
         ('(fun n : ℕ => (1 + 1 / ((n : ℝ) + 1)) ^ (n + 1))', '(fun n : ℕ => (3 * (n : ℝ) ^ 2 + 1) / ((n : ℝ) ^ 2 + 5))', False)]
for a, b, edge in pairs:
    La, Lb = lim(a), lim(b)
    both = limit(SEQ[a] / SEQ[b], n=oo)
    instances.append(O([a, b, str(La), str(Lb), '‹_›', '‹_›', '(by norm_num)'], both, La / Lb, edge=edge))
