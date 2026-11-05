# name: ra-poly-ratio-limit-zero-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.poly_ratio_limit_zero file "ra-poly-ratio-limit-zero-test.sage"
# description: Sage computes lim P(n)/Q(n) = 0 for deg P < deg Q
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
for P, Q, edge in [('x', 'x ** 2 + 1', False), ('1', 'x', True), ('x ** 2 - 3', 'x ** 3', False)]:
    lean = lambda s: s.replace('x', 'Polynomial.X').replace('**', '^')
    instances.append(O([f'({lean(P)} : Polynomial ℝ)', f'({lean(Q)} : Polynomial ℝ)', '(by compute_degree!)'],
                       limit(SR(P) / SR(Q), x=oo), QQ(0), edge=edge))
