# name: ra-zeta-even-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.zeta_even file "ra-zeta-even-test.sage"
# description: Sage's ζ(2k) (π²/6, π⁴/90, π⁶/945) against Euler's formula with B_2k (B_6 = 1/42)
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
for kk in [1, 2, 3]:
    formula = (-1) ** (kk + 1) * 2 ** (2 * kk - 1) * pi ** (2 * kk) * bernoulli(2 * kk) / factorial(2 * kk)
    instances.append(O([str(kk), '(by norm_num)'], zeta(2 * kk), formula, edge=bool(kk == 1)))
