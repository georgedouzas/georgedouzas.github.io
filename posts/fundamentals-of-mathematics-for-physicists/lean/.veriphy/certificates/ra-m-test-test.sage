# name: ra-m-test-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.m_test file "ra-m-test-test.sage"
# description: the M-test bound: Sage sums ∑ M_k (2^-k, 1/k², 0) and checks |f_k| ≤ M_k on a grid
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
k, x = var('k x')
instances = []
for f, M, edge in [(sin(k * x) / 2 ** k, QQ(1) / 2 ** k, False), (cos(k * x) / (k + 1) ** 2, 1 / (k + 1) ** 2, False), (SR(0), SR(0), True)]:
    Ms = sum(M, k, 0, oo) if M != 0 else SR(0)
    ok = all((abs(RI(f.subs(k=j, x=QQ(xv) / 7))) - RI(M.subs(k=j))).upper() <= 2 ** -100 for j in range(8) for xv in range(-7, 8))
    instances.append(O([f'(fun k x => {f})', f'(fun k => {M})', '‹_›', '‹_›'], B(ok and RI(Ms).is_finite() if hasattr(RI(Ms), 'is_finite') else ok), QQ(1), edge=edge))
