# name: ra-schwarzsymmetry-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.SchwarzSymmetry file "ra-schwarzsymmetry-test.sage"
# description: Sage's mixed partials agree for C² functions (x² y + sin y, e^(xy), x³y² + cos(x - y)) at sample points
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
for f, p, edge in [(x ** 2 * y + sin(y), (1, 2), False), (exp(x * y), (0, 1), False), (x ** 3 * y ** 2 + cos(x - y), (2, -1), False), (SR(0), (0, 0), True)]:
    a = diff(diff(f, x), y).subs(x=p[0], y=p[1]); b = diff(diff(f, y), x).subs(x=p[0], y=p[1])
    instances.append(O([f'(fun p : ℝ × ℝ => {f})', '(by fun_prop)', str(p[0]), str(p[1])], a, b, edge=edge))
