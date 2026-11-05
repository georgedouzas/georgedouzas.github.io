# name: ra-iteratedderiv-sq-exp-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.iteratedDeriv_sq_exp file "ra-iteratedderiv-sq-exp-test.sage"
# description: Sage's n-th derivative of x² eˣ against (x² + 2nx + n(n-1)) eˣ
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
instances = [O([str(n), str(v)], diff(x ** 2 * exp(x), x, n).subs(x=v), ((x ** 2 + 2 * n * x + n * (n - 1)) * exp(x)).subs(x=v), edge=bool(n == 0))
             for n, v in [(0, 1), (1, 2), (2, 0), (5, -1)]]
