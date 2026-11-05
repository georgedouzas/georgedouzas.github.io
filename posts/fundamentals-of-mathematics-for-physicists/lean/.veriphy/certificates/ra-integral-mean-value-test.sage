# name: ra-integral-mean-value-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.integral_mean_value file "ra-integral-mean-value-test.sage"
# description: the mean value point c of ∫ f = f(c)(b - a): Sage solves it (x² on [0, 1]: 1/√3; sin on [0, π]: arcsin(2/π))
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
for f, a, b, c, edge in [(x ** 2, 0, 1, 1 / sqrt(3), False), (sin(x), 0, pi, arcsin(2 / pi), False), (SR(5), 0, 1, SR(QQ(1) / 2), True)]:
    ok = bool((integrate(f, x, a, b) - f.subs(x=c) * (b - a)).simplify_full() == 0) and RI(a) <= RI(c) and RI(c) <= RI(b)
    instances.append(O([str(f), str(a), str(b), '(by norm_num)', '(by fun_prop)'], B(ok), QQ(1), edge=edge))
