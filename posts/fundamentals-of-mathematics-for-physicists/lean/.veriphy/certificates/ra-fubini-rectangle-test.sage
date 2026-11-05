# name: ra-fubini-rectangle-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.fubini_rectangle file "ra-fubini-rectangle-test.sage"
# description: Sage's iterated integrals in both orders agree on rectangles (x²y, e^(x+y), x sin y)
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
x, y = var('x y')
instances = []
for f, a, b, c, d, edge in [(x ** 2 * y, 0, 1, 0, 2, False), (exp(x + y), 0, 1, -1, 1, False), (x * sin(y), 0, 2, 0, pi, False), (SR(1), 0, 0, 0, 0, True)]:
    I1 = integrate(integrate(f, y, c, d), x, a, b); I2 = integrate(integrate(f, x, a, b), y, c, d)
    instances.append(O([f'(fun p : ℝ × ℝ => {f})', '(by fun_prop)', str(a), str(b), str(c), str(d), '(by norm_num)', '(by norm_num)'], I1, I2, edge=edge))
