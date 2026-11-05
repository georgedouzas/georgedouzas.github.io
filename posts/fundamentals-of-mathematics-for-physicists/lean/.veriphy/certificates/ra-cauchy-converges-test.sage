# name: ra-cauchy-converges-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.cauchy_converges file "ra-cauchy-converges-test.sage"
# description: Cauchy sequences converge: Sage computes the limit of the partial sums of ∑1/k!, ∑2^-k and of 1/n
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
n, k = var('n k')
instances = [O(['(fun n : ℕ => ∑ k ∈ Finset.range (n + 1), 1 / (k ! : ℝ))', '‹_›'], sum(1 / factorial(k), k, 0, oo), e),
             O(['(fun n : ℕ => ∑ k ∈ Finset.range (n + 1), (1 / 2 : ℝ) ^ k)', '‹_›'], sum(QQ(1) / 2 ** k, k, 0, oo), QQ(2)),
             O(['(fun _ : ℕ => (0 : ℝ))', '‹_›'], limit(0 * n, n=oo), QQ(0), edge=True)]
