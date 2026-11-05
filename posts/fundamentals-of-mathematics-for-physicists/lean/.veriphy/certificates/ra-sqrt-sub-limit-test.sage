# name: ra-sqrt-sub-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.sqrt_sub_limit file "ra-sqrt-sub-limit-test.sage"
# description: Sage's limit of √(n² + an) - n is a/2 (1/2 at a = 1)
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
n = var('n')
instances = [O([str(v), '(by norm_num)'], limit(sqrt(n ** 2 + v * n) - n, n=oo), SR(v) / 2, edge=bool(v == 0)) for v in [0, 1, 2, 5]]
