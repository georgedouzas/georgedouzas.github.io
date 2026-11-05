# name: ra-inverse-function-theorem-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.inverse_function_theorem file "ra-inverse-function-theorem-test.sage"
# description: Sage's derivative of the inverse (log for exp, √ for x² at 2, arcsin for sin at 0) is 1/f'(a)
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
for f, g, a, edge in [(exp(x), log(y), 0, False), (x ** 2, sqrt(y), 2, False), (sin(x), arcsin(y), 0, False), (x, y, 1, True)]:
    instances.append(O([str(f), str(a), str(diff(f, x).subs(x=a)), '‹_›', '(by norm_num)'], diff(g, y).subs(y=f.subs(x=a)), 1 / diff(f, x).subs(x=a), edge=edge))
