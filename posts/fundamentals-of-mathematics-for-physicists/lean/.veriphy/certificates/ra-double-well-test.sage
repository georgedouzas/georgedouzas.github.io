# name: ra-double-well-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.double_well file "ra-double-well-test.sage"
# description: Sage's V', V'' and the signs of V'' at 0 and ±v for sample λ, v
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
p = var('p')
instances = []
for lam, v, edge in [(1, 1, True), (2, 3, False), (QQ(1) / 2, -1, False)]:
    V = lam * (p ** 2 - v ** 2) ** 2 / 4
    ok = bool(diff(V, p) == lam * p * (p ** 2 - v ** 2)) or bool((diff(V, p) - lam * p * (p ** 2 - v ** 2)).expand() == 0)
    ok = ok and bool((diff(lam * p * (p ** 2 - v ** 2), p) - lam * (3 * p ** 2 - v ** 2)).expand() == 0)
    ok = ok and bool(diff(V, p, 2).subs(p=0) < 0) and bool(diff(V, p, 2).subs(p=v) > 0) and bool(V.subs(p=v) == 0)
    instances.append(O([str(lam), str(v), '(by norm_num)', '(by norm_num)'], B(ok), QQ(1), edge=edge))
