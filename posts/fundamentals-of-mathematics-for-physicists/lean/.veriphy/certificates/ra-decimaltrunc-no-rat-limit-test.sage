# name: ra-decimaltrunc-no-rat-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.decimalTrunc_no_rat_limit file "ra-decimaltrunc-no-rat-limit-test.sage"
# description: the truncations of √2, √3, √5 converge to the irrational number itself (no rational limit)
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
for p, edge in [(2, True), (3, False), (5, False)]:
    x = sqrt(p)
    close = abs(floor(x * 10 ** 30) / QQ(10 ** 30) - R(x)) < R(10) ** -29
    instances.append(O([f'√{p}', '(by norm_num)'], B(close and not QQ(p).is_square()), QQ(1), edge=edge))
