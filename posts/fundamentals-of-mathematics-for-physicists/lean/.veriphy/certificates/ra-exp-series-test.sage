# name: ra-exp-series-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.exp_series file "ra-exp-series-test.sage"
# description: Sage encloses ∑ x^k/k! rigorously (40 terms plus the tail bound) and compares with e^x by intervals
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
def exp_enclosure(v, N=40):
    S = sum(QQ(v) ** j / factorial(j) for j in range(N + 1))
    tail = abs(QQ(v)) ** (N + 1) / factorial(N + 1) * 2
    return RI(S - tail, S + tail)
instances = [O([str(v)], exp_enclosure(v), RI(exp(v)), edge=bool(v == 0)) for v in [0, 1, 2, -1, QQ(1) / 2]]
