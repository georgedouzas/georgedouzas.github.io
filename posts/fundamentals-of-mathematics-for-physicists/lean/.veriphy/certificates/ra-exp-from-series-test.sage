# name: ra-exp-from-series-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Exercises.exp_from_series file "ra-exp-from-series-test.sage"
# description: truncations (60 terms plus tail) of E(x)E(y) and E(x + y) agree with e^(x+y) by 256-bit intervals
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
def E(v, N=60):
    S = sum(QQ(v) ** j / factorial(j) for j in range(N + 1)); tail = 2 * abs(QQ(v)) ** (N + 1) / factorial(N + 1)
    return RI(S - tail, S + tail)
instances = []
for xv, yv, edge in [(0, 0, True), (1, 2, False), (QQ(-1) / 2, 3, False)]:
    ok = (E(xv) * E(yv)).overlaps(E(xv + yv)) and E(xv + yv).overlaps(RI(exp(xv + yv)))
    instances.append(O([str(xv), str(yv)], B(ok), QQ(1), edge=edge))
