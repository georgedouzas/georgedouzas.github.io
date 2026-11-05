# name: ra-stirling-ratio-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Exercises.stirling_ratio file "ra-stirling-ratio-test.sage"
# description: r_n = n!/(√(2πn)(n/e)ⁿ) ≥ 1 and decreasing, by 128-bit intervals (1.0844, 1.0084, 1.00083, … → 1)
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
def rr(n):
    return R(factorial(n)) / (R(2 * pi * n).sqrt() * (R(n) / R(e)) ** n)
instances = [O([str(n), '(by norm_num)'], B(rr(n) >= 1 and rr(n + 1) <= rr(n)), QQ(1), edge=bool(n == 1)) for n in [1, 10, 100, 1000]]
