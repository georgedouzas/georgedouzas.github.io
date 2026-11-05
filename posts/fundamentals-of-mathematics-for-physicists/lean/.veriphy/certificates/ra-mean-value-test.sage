# name: ra-mean-value-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.mean_value file "ra-mean-value-test.sage"
# description: the mean value point c (2/√3, π/2, log(e - 1), 0): Sage checks f'(c) = slope exactly and c ∈ (a, b) by intervals
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
instances = []
for f, a, b, c, edge in [(x ** 3, 0, 2, 2 / sqrt(3), False), (sin(x), 0, pi, pi / 2, False), (exp(x), 0, 1, log(e - 1), False),
                         (x ** 2, -1, 1, SR(0), True)]:
    slope = (f.subs(x=b) - f.subs(x=a)) / (b - a)
    ok = bool((diff(f, x).subs(x=c) - slope).simplify_full() == 0) and RI(a) < RI(c) and RI(c) < RI(b)
    instances.append(O([str(f), str(diff(f, x)), str(a), str(b), '(by norm_num)', '‹_›', '‹_›'], B(ok), QQ(1), edge=edge))
