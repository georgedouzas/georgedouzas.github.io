# name: ra-decimaltrunc-cauchy-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.decimalTrunc_cauchy file "ra-decimaltrunc-cauchy-test.sage"
# description: the decimal truncations of √2 (and of e, π) satisfy |a_m - a_n| < 10^-n, exactly
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
def trunc(x, n):
    return floor(x * 10 ** n) / QQ(10 ** n)
for x, xs, n, m, edge in [(sqrt(2), '√2', 0, 0, True), (sqrt(2), '√2', 1, 4, False), (sqrt(2), '√2', 3, 10, False),
                          (pi, 'π', 2, 7, False), (e, 'Real.exp 1', 5, 12, False)]:
    d = abs(trunc(x, m) - trunc(x, n))
    instances.append(O([xs, str(n), str(m), '(by norm_num)'], d, QQ(1) / 10 ** n, rel='<', edge=edge))
