# name: ra-log-factorial-asymptotic-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Exercises.log_factorial_asymptotic file "ra-log-factorial-asymptotic-test.sage"
# description: |log n! - (n log n - n)| ≤ log(n + 1) + 1 by 128-bit intervals
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
instances = [O([str(n), '(by norm_num)'], abs(R(log(factorial(n))) - (R(n) * R(log(n)) - n)), R(log(n + 1)) + 1, rel='<=', edge=bool(n == 1)) for n in [1, 5, 100, 10 ** 4]]
