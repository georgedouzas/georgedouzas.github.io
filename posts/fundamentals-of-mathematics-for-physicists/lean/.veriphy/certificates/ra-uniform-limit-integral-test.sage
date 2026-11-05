# name: ra-uniform-limit-integral-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.uniform_limit_integral file "ra-uniform-limit-integral-test.sage"
# description: Sage's lim ∫ f_n = ∫ lim f_n under uniform convergence on [a, b]
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
x, n = var('x n')
assume(n > 0)
instances = []
for f, g, a, b, edge in [(sin(n * x) / n, SR(0), 0, pi, False), (x + 1 / n, x, 0, 1, False), (x ** 2 * (1 + 1 / n), x ** 2, -1, 2, False), (SR(0) * n, SR(0), 0, 1, True)]:
    instances.append(O([f'(fun n x => {f})', str(g), str(a), str(b), '(by norm_num)', '‹_›', '‹_›'],
                       limit(integrate(f, x, a, b), n=oo), integrate(g, x, a, b), edge=edge))
forget()
