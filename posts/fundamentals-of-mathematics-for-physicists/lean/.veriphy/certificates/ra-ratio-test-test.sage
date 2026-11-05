# name: ra-ratio-test-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.ratio_test file "ra-ratio-test-test.sage"
# description: Sage computes the ratio limit L < 1 and the sum of |a_k| (x^k/k!, 10^k/k!, n!/n^n, n²/2^n, 2^n/n!)
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
for a, ex, edge in [('(fun k : ℕ => (1 : ℝ) / (k ! : ℝ))', 1 / factorial(k), True),
                    ('(fun k : ℕ => (10 : ℝ) ^ k / (k ! : ℝ))', 10 ** k / factorial(k), False),
                    ('(fun k : ℕ => ((k : ℝ) + 1) ^ 2 / 2 ^ (k + 1))', (k + 1) ** 2 / 2 ** (k + 1), False),
                    ('(fun k : ℕ => ((k + 1)! : ℝ) / ((k : ℝ) + 1) ^ (k + 1))', factorial(k + 1) / (k + 1) ** (k + 1), False)]:
    L = limit(abs(ex.subs(k=k + 1) / ex), k=oo)
    instances.append(O([a, str(L), '(by norm_num)', '‹_›', '‹_›'], B(bool(L < 1)), QQ(1), edge=edge))
