# name: fa-interval-kernel-limit-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.ChapterExercises.IntervalKernelLimit file "fa-interval-kernel-limit-test.sage"
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
instances = []
CASES = [(['1', '1', 'one_ne_zero'], True, {'K': '1', 'x': '1'}), (['3', '1/2', '_'], False, {'K': '3', 'x': '1/2'}), (['10', '-2', '_'], False, {'K': '10', 'x': '-2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('limit(1/(2*z_)*(sin(K*x)/sin(pi*x/(2*z_))), z_=oo)', scope))
    R = side(eval('sin(K*x)/(pi*x)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
