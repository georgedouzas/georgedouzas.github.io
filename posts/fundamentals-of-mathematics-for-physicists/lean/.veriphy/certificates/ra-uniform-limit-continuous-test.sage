# name: ra-uniform-limit-continuous-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.uniform_limit_continuous file "ra-uniform-limit-continuous-test.sage"
# description: the uniform limits g (0, x, x²) are within 1/n of f_n on a grid at n = 1000, and continuous
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
for f, g, edge in [(lambda n: sin(n * x) / n, SR(0), False), (lambda n: x + 1 / n, x, False), (lambda n: x ** 2 + exp(-n), x ** 2, False),
                   (lambda n: SR(0), SR(0), True)]:
    n = 1000
    close = all(abs(RI((f(n) - g).subs(x=QQ(j) / 10))) <= RI(1) / n + RI(10) ** -30 for j in range(-30, 31))
    instances.append(O([f'(fun n x => {f(SR.var("n"))})', str(g), '‹_›', '‹_›'], B(close and limit(g, x=0) == g.subs(x=0)), QQ(1), edge=edge))
