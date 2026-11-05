# name: ra-islub-iff-eps-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.isLUB_iff_eps file "ra-islub-iff-eps-test.sage"
# description: for intervals and finite sets, M = sup S exactly when M is an upper bound approached within every ε
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
instances = []
# S = [0, 1]: sup 1; M = 1 satisfies both sides, M = 2 neither (upper bound, but 2 - 1/2 is not exceeded)
for (sset, sup_s, M, edge) in [('Set.Icc (0 : ℝ) 1', 1, 1, True), ('Set.Icc (0 : ℝ) 1', 1, 2, False),
                               ('({1, 4, 2} : Set ℝ)', 4, 4, False), ('({1, 4, 2} : Set ℝ)', 4, 3, False)]:
    lhs = B(M == sup_s)
    rhs = B(M >= sup_s and M - sup_s == 0)   # the ε-condition: no gap between M and the elements
    instances.append(O([sset, str(M)], lhs, rhs, edge=edge))
