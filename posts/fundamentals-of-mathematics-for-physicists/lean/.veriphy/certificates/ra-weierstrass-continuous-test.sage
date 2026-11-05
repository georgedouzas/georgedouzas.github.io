# name: ra-weierstrass-continuous-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.weierstrass_continuous file "ra-weierstrass-continuous-test.sage"
# description: the M-test bound ∑ aⁿ = 1/(1 - a) Sage sums for the Weierstrass series
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
instances = []
for a, b, edge in [(QQ(1) / 2, 13, False), (QQ(1) / 3, 3, False), (QQ(1) / 2, 0, True)]:
    instances.append(O([str(a), str(b), '(by norm_num)', '(by norm_num)'], sum(SR(a) ** n, n, 0, oo), 1 / (1 - a), edge=edge))
