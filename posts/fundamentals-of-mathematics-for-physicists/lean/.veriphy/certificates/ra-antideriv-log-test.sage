# name: ra-antideriv-log-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.antideriv_log file "ra-antideriv-log-test.sage"
# description: Sage's derivative of log|x| is 1/x at sample points
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
x = var('x')
instances = [O([str(v), '(by norm_num)'], diff(log(abs(x)), x).subs(x=v), 1 / SR(v), edge=bool(v == 1)) for v in [1, -2, QQ(1) / 3]]
