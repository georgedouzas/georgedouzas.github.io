# name: ra-pow-not-uniform-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.pow_not_uniform file "ra-pow-not-uniform-test.sage"
# description: Sage's limits: xᵐ → 0 for 0 ≤ x < 1, and yⁿ → 1 as y → 1⁻ with yⁿ < 1 on [0, 1) (sup 1)
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
m, y = var('m y')
instances = []
for n, xv, edge in [(1, 0, True), (2, QQ(1) / 2, False), (5, QQ(9) / 10, False)]:
    ok = (xv == 0 or limit(SR(xv) ** m, m=oo) == 0) and limit(y ** n, y=1, dir='-') == 1
    instances.append(O([str(n), '(by norm_num)', str(xv), '(by norm_num)', '(by norm_num)'], B(ok), QQ(1), edge=edge))
