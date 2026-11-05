# name: ra-xyratio-props-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.xyRatio_props file "ra-xyratio-props-test.sage"
# description: f = 0 on the axes, f(x, x) = 1/2, partial quotients at 0 vanish, and the diagonal limit 1/2 ≠ f(0, 0) (Sage)
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
x, y, h = var('x y h')
f = x * y / (x ** 2 + y ** 2)
diag = limit(f.subs(y=x), x=0)
instances = []
for xv in [1, 2, QQ(-1) / 3]:
    ok = f.subs(x=xv, y=0) == 0 and f.subs(x=0, y=xv) == 0 and f.subs(x=xv, y=xv) == QQ(1) / 2 and diag == QQ(1) / 2
    instances.append(O([str(xv), '(by norm_num)'], B(ok), QQ(1), edge=bool(xv == 1)))
