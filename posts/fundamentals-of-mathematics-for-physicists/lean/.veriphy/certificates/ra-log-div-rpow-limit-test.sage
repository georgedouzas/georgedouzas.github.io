# name: ra-log-div-rpow-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.log_div_rpow_limit file "ra-log-div-rpow-limit-test.sage"
# description: Sage's limit of log x / x^r at ∞ is 0
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
instances = [O([str(r), '(by norm_num)'], limit(log(x) / x ** r, x=oo), QQ(0), edge=bool(r == 1)) for r in [1, QQ(1) / 2, 2]]
