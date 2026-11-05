# name: fa-torus-mode-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.CosetModes.torus_mode file "fa-torus-mode-test.sage"
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
z_ = var('z_'); y = var('y'); assume(y, 'real'); w_ = var('w_')
instances = []
CASES = [(['0', '0', '1', '1', '0', '0', 'one_ne_zero', 'one_ne_zero'], True, {'a': '0', 'b': '0', 'R': '1', 'S': '1', 'x': '0', 'y0': '0'}), (['1', '2', '1', '3', '1/2', '1', '_', '_'], False, {'a': '1', 'b': '2', 'R': '1', 'S': '3', 'x': '1/2', 'y0': '1'}), (['3', '-1', '2', '5', '-1', '2', '_', '_'], False, {'a': '3', 'b': '-1', 'R': '2', 'S': '5', 'x': '-1', 'y0': '2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('diff(cos(a*z_/R), z_, 2).subs(z_=x) + diff(cos(b*w_/S), w_, 2).subs(w_=y0)', scope))
    R = side(eval('-(a**2/R**2)*cos(a*x/R) - b**2/S**2*cos(b*y0/S)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
