# name: ra-exp-sub-one-div-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.exp_sub_one_div_limit file "ra-exp-sub-one-div-limit-test.sage"
# description: Sage's limit of (e^(cx) - 1)/x at 0 is c
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
instances = [O([str(c)], limit((exp(c * x) - 1) / x, x=0), SR(c), edge=bool(c == 0)) for c in [0, 1, 2, -3]]
