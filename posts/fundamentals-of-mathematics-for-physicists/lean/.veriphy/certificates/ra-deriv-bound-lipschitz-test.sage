# name: ra-deriv-bound-lipschitz-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.deriv_bound_lipschitz file "ra-deriv-bound-lipschitz-test.sage"
# description: |f(x) - f(y)| ≤ M |x - y| with M = sup |f'| (sin, cos with M = 1; x/2 + 3)
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
for f, M, edge in [(sin(x), 1, False), (cos(x), 1, False), (x / 2 + 3, QQ(1) / 2, True)]:
    pts = [(0, 1), (-2, QQ(1) / 3), (5, 7)]
    ok = all((abs(QQ(f.subs(x=a)) - QQ(f.subs(x=b))) <= M * abs(a - b)) if f.subs(x=a) in QQ and f.subs(x=b) in QQ
             else (abs(RI(f.subs(x=a)) - RI(f.subs(x=b))) <= M * abs(a - b)) for a, b in pts)
    instances.append(O([str(f), str(M), '‹_›', '‹_›', '0', '1'], B(ok), QQ(1), edge=edge))
