# name: ra-log-factorial-bounds-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Exercises.log_factorial_bounds file "ra-log-factorial-bounds-test.sage"
# description: n log n - n + 1 ≤ log n! ≤ (n+1) log(n+1) - n by 128-bit intervals
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
for n in [1, 2, 10, 100]:
    lf = R(log(factorial(n)))
    ok = R(n) * R(log(n)) - n + 1 <= lf and lf <= R(n + 1) * R(log(n + 1)) - (n + 1) + 1
    instances.append(O([str(n), '(by norm_num)'], B(ok), QQ(1), edge=bool(n == 1)))
