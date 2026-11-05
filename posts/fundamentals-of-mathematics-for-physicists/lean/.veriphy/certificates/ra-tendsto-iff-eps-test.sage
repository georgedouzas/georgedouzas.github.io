# name: ra-tendsto-iff-eps-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.tendsto_iff_eps file "ra-tendsto-iff-eps-test.sage"
# description: Sage computes the limit and, for ε = 1/10, the threshold N past which |a_n - L| < ε holds (checked on a window)
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
for a, edge in [('(fun _ : ℕ => (2 : ℝ))', True), ('(fun n : ℕ => 1 / ((n : ℝ) + 1))', False),
                ('(fun n : ℕ => (2 * (n : ℝ) + 1) / (n + 3))', False)]:
    L = lim(a)
    eps = QQ(1) / 10
    ok = all(abs(SEQ[a].subs(n=k) - L) < eps for k in range(60, 400))
    instances.append(O([a, str(L)], B(ok), QQ(1), edge=edge))
