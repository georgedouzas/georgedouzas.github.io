# name: fa-d1-deriv-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.MatrixCoefficients.d1_deriv file "fa-d1-deriv-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
CI = ComplexIntervalField(128)
def num(e):
    try:
        return RI(e)
    except (TypeError, ValueError):
        pass
    e = SR(e)
    for f in (lambda t: t, lambda t: t.simplify_full(), lambda t: t.canonicalize_radical()):
        try:
            return RI(f(e))
        except (TypeError, ValueError):
            pass
    try:
        return CI(e)
    except (TypeError, ValueError):
        return RI(e.n(prec=256))
def side(e):
    v = num(e)
    return v
z_ = var('z_'); y = var('y'); assume(y, 'real')
d1 = lambda t: matrix(SR, [[(1 + cos(t))/2, -(sqrt(2)/2)*sin(t), (1 - cos(t))/2], [sqrt(2)/2*sin(t), cos(t), -(sqrt(2)/2)*sin(t)], [(1 - cos(t))/2, sqrt(2)/2*sin(t), (1 + cos(t))/2]])
dp = lambda t: matrix(SR, [[-sin(t)/2, -(sqrt(2)/2)*cos(t), sin(t)/2], [sqrt(2)/2*cos(t), -sin(t), -(sqrt(2)/2)*cos(t)], [sin(t)/2, sqrt(2)/2*cos(t), -sin(t)/2]])
instances = []
CASES = [(['0', '0', '0'], True, {'t': '0', 'i': '0', 'j': '0'}), (['1', '0', '1'], False, {'t': '1', 'i': '0', 'j': '1'}), (['1/2', '2', '1'], False, {'t': '1/2', 'i': '2', 'j': '1'}), (['3', '1', '1'], False, {'t': '3', 'i': '1', 'j': '1'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('diff(d1(z_)[int(i), int(j)], z_).subs(z_=t)', scope))
    R = side(eval('dp(t)[int(i), int(j)]', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
