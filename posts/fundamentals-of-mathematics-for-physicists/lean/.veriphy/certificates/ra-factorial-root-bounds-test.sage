# name: ra-factorial-root-bounds-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.factorial_root_bounds file "ra-factorial-root-bounds-test.sage"
# description: 1/e ≤ (n!)^(1/n)/n ≤ (e√n)^(1/n)/e by 128-bit intervals at n = 1, 10, 100, 1000
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
for k in [1, 10, 100, 1000]:
    if k == 1:   # equality at n = 1: compare exactly
        ok = bool(exp(-1) <= 1) and bool((e * 1) * exp(-1) == 1)
    else:
        mid = R(factorial(k)) ** (R(1) / k) / k
        ok = R(exp(-1)) <= mid and mid <= (R(e) * R(k).sqrt()) ** (R(1) / k) * R(exp(-1))
    instances.append(O([str(k), '(by norm_num)'], B(ok), QQ(1), edge=bool(k == 1)))
