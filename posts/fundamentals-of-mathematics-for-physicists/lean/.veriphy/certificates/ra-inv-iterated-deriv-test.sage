# name: ra-inv-iterated-deriv-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.inv_iterated_deriv file "ra-inv-iterated-deriv-test.sage"
# description: Sage's (-1)ⁿ (d/da)ⁿ a⁻¹ = n! a^(-n-1)
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
a = var('a')
instances = [O([str(n), str(av), '(by norm_num)'], ((-1) ** n * diff(1 / a, a, n)).subs(a=av), factorial(n) * SR(av) ** (-n - 1), edge=bool(n == 0))
             for n, av in [(0, 1), (1, 2), (3, QQ(1) / 2), (5, -1)]]
