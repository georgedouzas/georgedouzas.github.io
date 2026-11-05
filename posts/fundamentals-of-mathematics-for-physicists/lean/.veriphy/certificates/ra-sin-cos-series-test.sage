# name: ra-sin-cos-series-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.sin_cos_series file "ra-sin-cos-series-test.sage"
# description: rigorous enclosures (30 terms plus tail) of the sin and cos series against sin x, cos x
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
def enclose(term, N, tail):
    # a rigorous enclosure of ∑ term(k): the partial sum up to N and the given bound on the tail
    S = sum(term(j) for j in range(N + 1))
    return RI(S - tail, S + tail)

instances = []
for x in [0, 1, 2, QQ(-1) / 2, 3]:
    xs = QQ(x)
    s = enclose(lambda j: (-1) ** j * xs ** (2 * j + 1) / factorial(2 * j + 1), 30, abs(xs) ** 63 / factorial(63))
    c = enclose(lambda j: (-1) ** j * xs ** (2 * j) / factorial(2 * j), 30, abs(xs) ** 62 / factorial(62))
    instances.append(O([str(x)], B(s.overlaps(RI(sin(xs))) and c.overlaps(RI(cos(xs)))), QQ(1), edge=bool(x == 0)))
