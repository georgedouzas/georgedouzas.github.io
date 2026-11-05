# name: ra-sin-terms-pi-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.sin_terms_pi file "ra-sin-terms-pi-test.sage"
# description: π^(2m+1)/(2m+1)! < 10⁻⁶ iff m ≥ 8 by 128-bit intervals (m = 7: 2.2 · 10⁻⁵, m = 8: 8.0 · 10⁻⁷)
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
for m in [0, 7, 8, 12]:
    v = R(pi) ** (2 * m + 1) / factorial(2 * m + 1)
    instances.append(O([str(m)], B(v < R(10) ** -6), B(m >= 8), edge=bool(m == 0)))
