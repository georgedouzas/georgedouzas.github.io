# name: ra-integral-rules-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.integral_rules file "ra-integral-rules-test.sage"
# description: Sage's exact integrals satisfy linearity, additivity and orientation reversal
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
for f, g, a, b, c, k, edge in [(x ** 2, sin(x), 0, 1, 2, 3, False), (exp(x), cos(x), -1, 0, 1, -2, False), (SR(1), SR(0), 0, 0, 0, 0, True)]:
    lin = bool((integrate(f + k * g, x, a, b) - integrate(f, x, a, b) - k * integrate(g, x, a, b)).simplify_full() == 0)
    add = bool((integrate(f, x, a, b) + integrate(f, x, b, c) - integrate(f, x, a, c)).simplify_full() == 0)
    rev = bool((integrate(f, x, a, b) + integrate(f, x, b, a)).simplify_full() == 0)
    instances.append(O([str(f), str(g), str(a), str(b), str(c), str(k), '(by fun_prop)', '(by fun_prop)'], B(lin and add and rev), QQ(1), edge=edge))
