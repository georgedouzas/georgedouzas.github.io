# name: ra-one-add-div-pow-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.one_add_div_pow_limit file "ra-one-add-div-pow-limit-test.sage"
# description: Sage's symbolic limit of (1 + x/n)^n is e^x (e at x = 1)
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
instances = [O([str(v)], limit((1 + v / n) ** n, n=oo), exp(v), edge=bool(v == 0)) for v in [0, 1, 2, -1, QQ(1) / 2]]
